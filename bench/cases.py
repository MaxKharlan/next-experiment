"""Benchmark cases: a finished fine-tuning job, what a user would say about it, the truth.

A case is bench/cases/<split>/<id>.json. The methods see only its "case" part, the
same input the agent starts from: the symptom, the task, the job's config, the loss
per checkpoint and the findings. "truth" lists the injected fault classes and stays
with the scorer; a healthy job has none. Dev cases tune prompts and thresholds;
held-out cases stay closed until the final run (section 10 of the project doc).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CASES_DIR = Path(__file__).parent / "cases"


@dataclass
class BenchCase:
    id: str
    split: str  # dev or heldout
    task: str  # e.g. banking77; its reference is bench/references/<task>.json
    model: str
    truth: list[str]  # injected fault classes; empty for a healthy job
    hyperparameters: dict[str, Any]  # the job's, in Token Factory names
    case: dict[str, Any]  # what the methods see
    train: str
    valid: str
    heldout: str
    job_id: str = "local"  # the job on Token Factory, the failed model
    heldout_metric: float | None = None  # the job's exact match on the held-out set
    tokens_per_example: float = 57  # chat template tokens, for prices
    # saved investigations whose causal checks and retrain seed the caches
    reports: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.truth

    @property
    def train_examples(self) -> int:
        lines = Path(self.train).read_text(encoding="utf-8").splitlines()
        return sum(1 for line in lines if line.strip())

    def job_config(self) -> dict[str, Any]:
        return {
            "hyperparameters": self.hyperparameters,
            "checkpoints": self.case.get("checkpoints", []),
        }

    def save(self, root: Path = CASES_DIR) -> Path:
        path = root / self.split / f"{self.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path


def load_cases(split: str, root: Path = CASES_DIR) -> list[BenchCase]:
    return [
        BenchCase(**json.loads(p.read_text(encoding="utf-8")))
        for p in sorted((root / split).glob("*.json"))
    ]
