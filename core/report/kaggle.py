"""Offline analysis of declared Kaggle measurements; no training or inference."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.interventions.stats import causal_check


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Measurement(Record):
    run_id: str = Field(min_length=1)
    role: Literal["healthy", "damaged", "target", "control"]
    seed: int
    metric: float = Field(ge=0, le=1)
    train_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hyperparameters: dict
    summary_path: str | None = None


class Evaluation(Record):
    split: Literal["check", "validation", "test"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    examples: int = Field(gt=0)
    metric: Literal["exact_match"] = "exact_match"


class KaggleResults(Record):
    version: Literal[1]
    backend: Literal["kaggle"]
    model: str = Field(min_length=1)
    fault: Literal["answer_truncation"]
    evaluation: Evaluation
    runs: list[Measurement] = Field(min_length=1)
    # These are declarations from the experiment, not inferred from score values.
    control_description: str = ""
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_runs(self):
        ids = [run.run_id for run in self.runs]
        keys = [(run.role, run.seed) for run in self.runs]
        if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
            raise ValueError("Duplicate run ID or role/seed")
        return self


def analyze(path: Path, *, min_effect: float, noise_sd: float | None = None) -> dict:
    """Use the existing causal rule only for a complete matched-seed check pair."""
    if not math.isfinite(min_effect) or not 0 < min_effect <= 1:
        raise ValueError("min_effect must be finite and in (0, 1]")
    if noise_sd is not None and (not math.isfinite(noise_sd) or noise_sd <= 0):
        raise ValueError("noise_sd must be finite and positive")
    path = Path(path)
    source = path.read_bytes()
    bundle = KaggleResults.model_validate_json(source)
    groups = {}
    sources = []
    used_paths = set()
    for run in bundle.runs:
        groups.setdefault(run.role, []).append(run)
        if run.summary_path is None:
            continue
        summary_path = (path.parent / run.summary_path).resolve(strict=True)
        if summary_path in used_paths:
            raise ValueError("A summary file cannot represent multiple runs")
        used_paths.add(summary_path)
        raw = summary_path.read_bytes()
        summary = json.loads(raw)
        if (
            summary["model"] != bundle.model
            or summary["seed"] != run.seed
            or summary["hyperparameters"] != run.hyperparameters
            or summary["final"]["exact_match"] != run.metric
        ):
            raise ValueError(f"Summary does not match run {run.run_id}")
        # train_lora.py evaluates on validation. An external evaluator must
        # explicitly record a different split and its fingerprint in its summary.
        evaluation = summary.get("evaluation")
        if evaluation is None:
            if bundle.evaluation.split != "validation":
                raise ValueError("Legacy summary scores validation, not check/test")
            if summary["valid_examples"] != bundle.evaluation.examples:
                raise ValueError("Validation example count does not match")
        elif Evaluation.model_validate(evaluation) != bundle.evaluation:
            raise ValueError("Summary evaluation does not match the bundle")
        sources.append(
            {"path": str(summary_path), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    for runs in groups.values():
        runs.sort(key=lambda run: run.seed)
    result = {
        "version": 1,
        "backend": "kaggle",
        "model": bundle.model,
        "fault": bundle.fault,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_files": sources,
        "provenance": "Run roles and dataset hashes are declared by the exporter",
        "evaluation": bundle.evaluation.model_dump(),
        "environment": bundle.environment,
        "control_description": bundle.control_description,
        "runs": [run.model_dump() for run in bundle.runs],
        "means": {
            role: statistics.fmean(r.metric for r in runs)
            for role, runs in groups.items()
        },
        "min_effect": min_effect,
        "noise_sd": noise_sd,
        "noise_source": "caller supplied"
        if noise_sd is not None
        else "estimated from these arms",
        "new_training_jobs": 0,
        "new_api_calls": 0,
        "causal_result": None,
        "status": "insufficient_evidence",
    }
    if {"healthy", "damaged"} <= groups.keys():
        result["quality_drop"] = result["means"]["healthy"] - result["means"]["damaged"]
    if not {"target", "control"} <= groups.keys():
        result["reason"] = (
            "Target and control measurements are required for a causal verdict"
        )
        return result
    if bundle.evaluation.split != "check":
        result["reason"] = "Causal selection requires a separate check set"
        return result
    target, control = groups["target"], groups["control"]
    if [r.seed for r in target] != [r.seed for r in control]:
        raise ValueError("Target and control must contain the same seeds")
    if len(target) < 2:
        result["reason"] = "At least two seeds per arm are required"
        return result
    # The current permutation implementation enumerates combinations in memory.
    if len(target) > 8:
        raise ValueError(
            "Current offline analyzer supports at most eight seeds per arm"
        )
    if not bundle.control_description.strip():
        raise ValueError("Describe the control intervention")
    hp = target[0].hyperparameters
    if any(run.hyperparameters != hp for run in target + control):
        raise ValueError("Answer-cut arms must use identical hyperparameters")
    if any(
        t.train_sha256 == c.train_sha256 for t, c in zip(target, control, strict=True)
    ):
        raise ValueError("Target and control training datasets must differ")
    check = causal_check(
        [run.metric for run in target],
        [run.metric for run in control],
        min_effect=min_effect,
        noise_sd=noise_sd,
    )
    result.update(status=check.status, reason=check.reason, causal_result=asdict(check))
    result["decision_rule"] = "core.interventions.stats.causal_check"
    result["permutation_can_reach_0_05"] = (
        math.comb(2 * len(target), len(target)) * 0.05 >= 1
    )
    return result
