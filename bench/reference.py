"""B5: the reference model of a task, the level a fixed model should reach.

For a task and a base model a small grid of healthy recipes trains on Token Factory
on the task's training file. The recipe with the best exact match on the validation
file becomes the reference, and its exact match on the held-out set is the bar for
"fix success" in the benchmark (section 10 of the project doc). The recipe is chosen
on validation and scored on held-out, so the bar is not picked on the data it is
measured on. Job IDs are saved under <prefix>/jobs.json as they are created, so a
rerun waits for the same jobs. The recipes change base_hp, the healthy recipe by
default; the benchmark also trains a method's fix this way, on the case's recipe.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.faults.recipe import HEALTHY_HP
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TokenFactory
from core.interventions.tf_runner import (
    TF_LORA,
    download_last_checkpoint,
    wait_for_jobs,
)
from core.planner.job_cost import estimate_job

# recipe name -> changes to HEALTHY_HP; LoRA r16 and alpha 16 stay
GRID: dict[str, dict[str, Any]] = {
    "lr1e-4-e3": {"learning_rate": 1e-4, "n_epochs": 3},
    "lr2e-4-e3": {"learning_rate": 2e-4, "n_epochs": 3},
    "lr1e-4-e6": {"learning_rate": 1e-4, "n_epochs": 6},
    "lr2e-4-e6": {"learning_rate": 2e-4, "n_epochs": 6},
}


def grid_cost(
    model: str,
    examples: int,
    tokens_per_example: float,
    grid: dict[str, dict[str, Any]] = GRID,
) -> float:
    return sum(
        estimate_job(
            model, examples, tokens_per_example, {**HEALTHY_HP, **patch}["n_epochs"]
        ).cost_usd
        for patch in grid.values()
    )


@dataclass
class ReferenceRun:
    tf: TokenFactory
    storage: LocalStorage
    model: str
    train_path: Path
    valid_path: Path
    heldout_path: Path
    # (adapter directory, data file) -> exact match
    score: Callable[[Path, Path], float]
    prefix: str
    grid: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(GRID))
    base_hp: dict[str, Any] = field(default_factory=lambda: dict(HEALTHY_HP))
    poll_s: float = 30
    timeout_s: float = 6 * 3600
    sleep: Callable[[float], None] = time.sleep

    def __call__(self) -> dict[str, Any]:
        state_key = f"{self.prefix}/jobs.json"
        state = (
            json.loads(self.storage.get_text(state_key))
            if self.storage.exists(state_key)
            else {}
        )
        for key, path in (("_train", self.train_path), ("_valid", self.valid_path)):
            if key not in state:
                state[key] = self.tf.upload(path)
        for name, patch in self.grid.items():
            if name not in state:
                state[name] = self.tf.create_job(
                    self.model,
                    state["_train"],
                    validation_file=state["_valid"],
                    hyperparameters={**self.base_hp, **patch, **TF_LORA},
                    suffix=f"ne-ref-{name}",
                )
                print(f"{name}: submitted {state[name]}", flush=True)
            self.storage.put_text(state_key, json.dumps(state, indent=2))

        jobs = {name: state[name] for name in self.grid}
        finished = wait_for_jobs(
            self.tf,
            set(jobs.values()),
            poll_s=self.poll_s,
            timeout_s=self.timeout_s,
            sleep=self.sleep,
        )
        for job_id, job in finished.items():
            if job.status != "succeeded":
                raise RuntimeError(f"{job_id} ended {job.status}: {job.error}")
        adapters = {
            name: download_last_checkpoint(self.tf, self.storage, job_id, self.prefix)
            for name, job_id in jobs.items()
        }
        valid = {name: self.score(adapters[name], self.valid_path) for name in jobs}
        for name, score in valid.items():
            print(f"{name}: {score:.3f} on {self.valid_path}", flush=True)
        best = max(valid, key=valid.get)
        heldout = self.score(adapters[best], self.heldout_path)
        rows = self.heldout_path.read_text(encoding="utf-8").splitlines()
        return {
            "model": self.model,
            "recipe": best,
            "hyperparameters": {**self.base_hp, **self.grid[best]},
            "job_id": jobs[best],
            "valid_exact_match": valid[best],
            "heldout_exact_match": heldout,
            "heldout_examples": sum(1 for r in rows if r.strip()),
            "grid": {
                name: {"job_id": jobs[name], "valid_exact_match": valid[name]}
                for name in jobs
            },
        }
