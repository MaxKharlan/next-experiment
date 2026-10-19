"""Validate an explicit investigation case before starting a worker."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field

from apps.api.live_protocol import ProtocolConfig, prepare_live_run, verify_datasets
from core.case import InvestigationCase
from core.connectors.token_factory import hyperparameter_problem
from core.eval.data import read_examples
from core.eval.runtime import EvaluationRuntime
from core.eval.structured import validate_answer, validate_evaluator
from core.probes.level0.lengths import read_chat_jsonl


class ExperimentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1)
    noise_sd: float = Field(gt=0)
    min_effect: float = Field(gt=0)
    tokens_per_example: float = Field(gt=0)
    seeds: list[int] = Field(default_factory=lambda: [0, 1], min_length=2)
    evaluation_runtime: EvaluationRuntime | None = None


class CaseStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: InvestigationCase
    experiment: ExperimentSettings


def prepare_case(request: CaseStart) -> dict:
    case = request.case
    if case.mode != "A":
        raise ValueError("Explicit cases currently support Token Factory mode A")
    if case.job.model != "Qwen/Qwen3-0.6B":
        raise ValueError("Supported model: Qwen/Qwen3-0.6B")
    evaluator = validate_evaluator(case.task, case.evaluator)
    if not math.isfinite(case.budget_usd):
        raise ValueError("The budget must be finite")

    hp = case.job.hyperparameters
    required = {
        "learning_rate",
        "n_epochs",
        "lora_r",
        "lora_alpha",
        "batch_size",
        "context_length",
    }
    if missing := required - hp.keys():
        raise ValueError(f"Missing explicit hyperparameters: {sorted(missing)}")
    for key, value in hp.items():
        if key in {"lora", "packing"}:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be a finite number")
    for key in ("n_epochs", "lora_r", "batch_size", "context_length"):
        if type(hp[key]) is not int or hp[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if hp["learning_rate"] <= 0 or hp["context_length"] < 8192:
        raise ValueError("Require positive learning_rate and context_length >= 8192")
    if hp.get("lora", True) is not True or hp.get("packing", False) is not False:
        raise ValueError("The current runner requires LoRA without packing")
    if problem := hyperparameter_problem(hp):
        raise ValueError(problem)

    refs = {
        role: getattr(case.datasets, role)
        for role in ("train", "validation", "check", "test")
    }
    for role, ref in refs.items():
        path = Path(ref.ref)
        rows = read_chat_jsonl(path)
        for line, messages in rows:
            location = f"{role}, line {line}"
            for message in messages:
                if message["role"] not in {"system", "user", "assistant"}:
                    raise ValueError(f"{location}: unsupported message role")
                if not message["content"].strip():
                    raise ValueError(f"{location}: message content must not be blank")
            if messages[-1]["role"] != "assistant":
                raise ValueError(
                    f"{location}: example must end with an assistant answer"
                )
            if not any(message["role"] == "user" for message in messages[:-1]):
                raise ValueError(f"{location}: example must contain a user prompt")
        if any(not line.strip() for line in path.read_text().splitlines()):
            raise ValueError(f"{role}: blank JSONL lines are not supported")

    # Damaged training labels remain inspectable; scoring references must be valid.
    if case.task != "banking77":
        for role in ("check", "test"):
            for row in read_examples(refs[role].ref):
                validate_answer(case.task, row.expected)

    if case.task != "banking77":
        from core.eval.audit_case import audit_case

        audit_case(case)

    config = ProtocolConfig(
        **request.experiment.model_dump(),
        model=case.job.model,
        check=refs["check"].ref,
        budget_usd=case.budget_usd,
    )
    source = SimpleNamespace(
        model=case.job.model,
        job_id=case.job.id,
        hyperparameters=hp,
        train=refs["train"].ref,
        valid=refs["validation"].ref,
        heldout=refs["test"].ref,
    )
    protocol = prepare_live_run(source, config=config)
    for role, ref in refs.items():
        actual = protocol["datasets"][role]
        if ref.sha256 is not None and ref.sha256 != actual["sha256"]:
            raise ValueError(f"{role}: declared SHA-256 does not match")
        if ref.examples is not None and ref.examples != actual["examples"]:
            raise ValueError(f"{role}: declared example count does not match")

    protocol["evaluator"] = evaluator.model_dump()
    protocol["task"] = case.task
    protocol["calibration_scope"] = "caller-specified experimental settings"
    verify_datasets(protocol)

    # Evaluation datasets and any benchmark truth stay out of model prompts.
    agent_case = {
        "task": case.task,
        "symptom": case.symptom,
        "job": case.job.model_dump(exclude={"checkpoints"}),
        "checkpoints": case.job.checkpoints,
    }
    return {"agent_case": agent_case, "protocol": protocol}


def verify_original_job(tf, prepared):
    """Read the original job before any hypothesis calls or new training."""
    protocol = prepared["protocol"]
    job = tf.job(protocol["job_id"])
    if job.status != "succeeded":
        raise ValueError("The original job must have status succeeded")
    if job.model != protocol["model"]:
        raise ValueError("Original job model does not match the case")
    actual = job.hyperparameters
    if hasattr(actual, "model_dump"):
        actual = actual.model_dump()
    if not isinstance(actual, dict):
        raise TypeError("Original job hyperparameters are unavailable")
    for key, expected in protocol["hyperparameters"].items():
        if actual.get(key) != expected:
            raise ValueError(f"Original job hyperparameter mismatch: {key}")
