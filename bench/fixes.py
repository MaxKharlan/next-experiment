"""Fix success: does the model retrained with a method's fix reach the reference?

A method that confirms a cause gets the standard fix for that fault class: the target
of its causal check (core/interventions/pairs.py), applied to the case's recipe and
trained on the whole training file. Methods that confirm the same cause on the same
case get the same fix, so the retrained metric is cached by the fixed
hyperparameters and trained once. A fix succeeds when its held-out metric reaches
the reference of the task and model (bench/references, B5) within TOLERANCE, which
is set here, before any run (section 10 of the project doc). Token Factory training
is deterministic, so a fix whose recipe is the reference's own scores exactly the
reference: seed_reference adds it without a new job.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.cases import BenchCase
from core.agent.context import InvestigationContext
from core.agent.schemas import Hypothesis
from core.connectors.storage import Storage
from core.interventions.pairs import BUILDERS, make_pair

# absolute exact match; one standard error on 200 held-out queries is about 0.035
TOLERANCE = 0.05
REFERENCES_DIR = Path(__file__).parent / "references"


def fixed_hyperparameters(case: BenchCase, cause: str | None) -> dict[str, Any] | None:
    """The case's recipe with the standard fix for cause, or None without one."""
    if cause not in BUILDERS:
        return None
    hypothesis = Hypothesis(
        id=cause, fault_class=cause, location="", prior=0.0, rationale=""
    )
    ctx = InvestigationContext(case.id, "A", case.job_id, case.job_config())
    try:
        target, _ = make_pair(hypothesis, ctx)
    except ValueError:  # e.g. the learning rate is already at the top of the range
        return None
    return target.hyperparameters(case.hyperparameters)


def reference_metric(
    task: str, model: str, root: Path = REFERENCES_DIR
) -> float | None:
    path = root / f"{task}.json"
    if not path.exists():
        return None
    found = json.loads(path.read_text(encoding="utf-8")).get(model)
    return found["heldout_exact_match"] if found else None


def fix_key(hyperparameters: dict[str, Any]) -> str:
    return json.dumps(hyperparameters, sort_keys=True)


@dataclass
class FixCache:
    storage: Storage
    key: str  # storage key of the cache file
    # fixed hyperparameters -> (held-out metric, job ID); None: never trains
    train: Callable[[dict[str, Any]], tuple[float, str]] | None = None

    def load(self) -> dict[str, Any]:
        if not self.storage.exists(self.key):
            return {}
        return json.loads(self.storage.get_text(self.key))

    def add(self, hyperparameters: dict[str, Any], metric: float, job_id: str) -> None:
        cache = self.load()
        cache[fix_key(hyperparameters)] = {"heldout": metric, "job_id": job_id}
        self.storage.put_text(self.key, json.dumps(cache, indent=2))

    def __call__(self, hyperparameters: dict[str, Any]) -> float | None:
        found = self.load().get(fix_key(hyperparameters))
        if found is not None:
            return found["heldout"]
        if self.train is None:
            return None
        metric, job_id = self.train(hyperparameters)
        self.add(hyperparameters, metric, job_id)
        return metric


def seed_fixes(cache: FixCache, report_path: Path) -> int:
    """Add the retrain of a saved investigation (scripts/retrain_tf.py)."""
    if not report_path.exists():
        return 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    steps = {s["kind"]: s for s in report["steps"]}
    if "retrain" not in steps or "comparison" not in steps:
        return 0
    fixed = next(m for m in steps["comparison"]["models"] if m["name"] == "fixed")
    cache.add(steps["retrain"]["hyperparameters"], fixed["score"], fixed["job_id"])
    return 1


def seed_reference(
    cache: FixCache, case: BenchCase, root: Path = REFERENCES_DIR
) -> int:
    """Add the reference job of the case's task and model, if it trained on as many
    examples as the case."""
    path = root / f"{case.task}.json"
    if not path.exists():
        return 0
    found = json.loads(path.read_text(encoding="utf-8")).get(case.model)
    if not found or found.get("train_examples") != case.train_examples:
        return 0
    cache.add(found["hyperparameters"], found["heldout_exact_match"], found["job_id"])
    return 1
