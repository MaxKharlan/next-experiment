"""Validate and freeze the existing Token Factory experiment protocol."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.planner.job_cost import estimate_job

CONFIG_PATH = Path("config/live-tf-banking77.json")


class ProtocolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str
    model: str
    check: str
    noise_sd: float = Field(gt=0)
    min_effect: float = Field(gt=0)
    seeds: list[int] = Field(min_length=2)
    tokens_per_example: float = Field(gt=0)
    budget_usd: float = Field(gt=0)

    @field_validator("seeds")
    @classmethod
    def unique_seeds(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Seeds must be unique")
        return value


def inspect_datasets(paths):
    """Reject missing files, aliased paths and identical cross-split records."""
    resolved = {role: Path(path).resolve(strict=True) for role, path in paths.items()}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("Dataset splits must use distinct files")

    manifests = {}
    owners = {}
    for role, path in resolved.items():
        raw = path.read_bytes()
        records = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON in {role}, line {line_number}") from err
            if not isinstance(record, dict) or not isinstance(
                record.get("messages"), list
            ):
                raise TypeError(
                    f"{role}, line {line_number}: expected a chat messages list"
                )
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            previous = owners.get(digest)
            if previous is not None and previous != role:
                raise ValueError(
                    f"Identical record appears in both {previous} and {role}"
                )
            owners[digest] = role
            records.append(record)

        if not records:
            raise ValueError(f"Empty dataset: {role}")
        manifests[role] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "examples": len(records),
        }
    return manifests


def verify_datasets(protocol):
    saved = protocol["datasets"]
    current = inspect_datasets({role: entry["path"] for role, entry in saved.items()})
    if current != saved:
        raise ValueError("Dataset files changed after the investigation started")


def prepare_live_run(case, config_path=CONFIG_PATH, *, config=None):
    if config is None:
        config = ProtocolConfig.model_validate_json(Path(config_path).read_text())
    if case.model != config.model:
        raise ValueError("The case model does not match the calibrated protocol")
    if not case.job_id.startswith("ftjob-"):
        raise ValueError("The live TF protocol requires the original job ID")

    datasets = inspect_datasets(
        {
            "train": case.train,
            "validation": case.valid,
            "check": config.check,
            "test": case.heldout,
        }
    )
    hp = dict(case.hyperparameters)
    epochs = hp.get("n_epochs")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("n_epochs must be a positive integer")

    one = estimate_job(
        case.model,
        datasets["train"]["examples"],
        config.tokens_per_example,
        epochs,
        hp.get("context_length", 8192),
    )
    return {
        "version": 1,
        "name": config.name,
        "model": case.model,
        "job_id": case.job_id,
        "hyperparameters": hp,
        "datasets": datasets,
        "noise_sd": config.noise_sd,
        "min_effect": config.min_effect,
        "seeds": config.seeds,
        "tokens_per_example": config.tokens_per_example,
        "budget_usd": config.budget_usd,
        # Conservative planning estimate for the current recipe interventions.
        "pair_estimate_usd": 2 * len(config.seeds) * one.cost_usd,
        "cost_basis": "estimated trained tokens; not a provider invoice",
    }


def estimate_pair_cost(protocol, target, control):
    """Estimate each arm using its actual recipe and data fraction."""
    total = 0.0
    for arm in (target, control):
        fraction = arm.data_fraction
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("data_fraction must be in (0, 1]")
        examples = max(1, int(protocol["datasets"]["train"]["examples"] * fraction))
        hp = arm.hyperparameters(protocol["hyperparameters"])
        estimate = estimate_job(
            protocol["model"],
            examples,
            protocol["tokens_per_example"],
            hp["n_epochs"],
            hp.get("context_length", 8192),
        )
        total += len(protocol["seeds"]) * estimate.cost_usd
    return total


def main():
    from bench.cases import load_cases

    case = next(c for c in load_cases("dev") if c.id == "fault6-lr1e-5")
    print(json.dumps(prepare_live_run(case), indent=2))


if __name__ == "__main__":
    main()
