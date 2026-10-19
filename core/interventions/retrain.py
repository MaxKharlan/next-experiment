"""B18: the full retrain with the confirmed fix and the comparison of three models.

A confirmed verdict comes with the target's patch to the failed job's
hyperparameters. The retrain applies it to the whole training file as one
Post-training job. It is the only level-3 step: the agent proposes it with its price
and starts it only after the user agrees. Then the base model, the failed job and the
retrained job answer the same held-out set.

The failed model is the user's job when the investigation has one. A case without
one, like the demo cases, uses the control arm of the confirmed check when that arm
is the failed recipe again. Otherwise the failed recipe trains as a second job next
to the retrain. Scoring is injected as in tf_runner, so this module needs no torch.
Job IDs are saved under <prefix>/retrain.json as they are created, so a rerun waits
for the same jobs instead of paying for new ones.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TokenFactory
from core.interventions.pairs import Intervention
from core.interventions.tf_runner import (
    TF_LORA,
    download_last_checkpoint,
    wait_for_jobs,
)
from core.planner.job_cost import estimate_job


@dataclass
class RetrainPlan:
    hypothesis_id: str
    model: str
    base_hyperparameters: dict[str, Any]  # the failed job's
    patch: dict[str, Any]  # the target's changes, without data_fraction
    diff: str
    train_examples: int
    tokens: int
    cost_usd: float
    seconds: float  # without the wait for GPU capacity
    # the job that stands for the failed model; None trains the failed recipe too
    failed_job_id: str | None = None
    jobs: list[str] = field(default_factory=lambda: ["fixed"])
    data_patch: dict[str, Any] | None = None

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {**self.base_hyperparameters, **self.patch}


def failed_job(job_id: str, control: Intervention) -> str | None:
    """The user's job, or a control job that reran the failed recipe unchanged."""
    if job_id.startswith("ftjob-"):
        return job_id
    if not control.patch and control.ft_job_ids:
        return control.ft_job_ids[0]
    return None


def plan_retrain(
    target: Intervention,
    base_hp: dict[str, Any],
    model: str,
    train_examples: int,
    tokens_per_example: float,
    failed_job_id: str | None,
) -> RetrainPlan:
    fixed_hp = target.hyperparameters(base_hp)
    context = fixed_hp.get("context_length", 8192)
    estimates = [
        estimate_job(
            model, train_examples, tokens_per_example, fixed_hp["n_epochs"], context
        )
    ]
    jobs = ["fixed"]
    if failed_job_id is None:
        estimates.append(
            estimate_job(
                model, train_examples, tokens_per_example, base_hp["n_epochs"], context
            )
        )
        jobs.append("failed")
    return RetrainPlan(
        hypothesis_id=target.hypothesis_id,
        model=model,
        base_hyperparameters=dict(base_hp),
        patch={k: v for k, v in target.patch.items() if k != "data_fraction"},
        diff=target.diff(base_hp),
        train_examples=train_examples,
        tokens=sum(e.tokens for e in estimates),
        cost_usd=sum(e.cost_usd for e in estimates),
        seconds=max(e.seconds for e in estimates),  # the jobs run side by side
        failed_job_id=failed_job_id,
        jobs=jobs,
    )


@dataclass
class Retrainer:
    tf: TokenFactory
    storage: LocalStorage
    train_path: Path
    valid_path: Path
    heldout_path: Path
    # (adapter directory, data file) -> metric
    score_adapter: Callable[[Path, Path], float]
    # (base model, data file, directory with the chat template to prompt with)
    score_base: Callable[[str, Path, Path], float]
    prefix: str
    metric: str = "exact_match"
    heldout_note: str = ""
    poll_s: float = 30
    timeout_s: float = 6 * 3600
    sleep: Callable[[float], None] = time.sleep

    def __call__(
        self,
        plan: RetrainPlan,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Train, score the three models and return the comparison step."""
        if plan.data_patch is not None:
            raise ValueError("Data patches require the provenance-bound data retrainer")
        state_key = f"{self.prefix}/retrain.json"
        state = (
            json.loads(self.storage.get_text(state_key))
            if self.storage.exists(state_key)
            else {}
        )
        for key, path in (("_train", self.train_path), ("_valid", self.valid_path)):
            if key not in state:
                state[key] = self.tf.upload(path)
                self.storage.put_text(state_key, json.dumps(state, indent=2))
        recipes = {"fixed": plan.hyperparameters, "failed": plan.base_hyperparameters}
        for name in plan.jobs:
            if name not in state:
                attempts = state.setdefault("_attempts", {})
                if name in attempts:
                    raise ValueError(
                        "Unresolved retrain submission; reconcile before retrying"
                    )
                attempts[name] = "submitting"
                self.storage.put_text(state_key, json.dumps(state, indent=2))
                state[name] = self.tf.create_job(
                    plan.model,
                    state["_train"],
                    validation_file=state["_valid"],
                    hyperparameters={**recipes[name], **TF_LORA},
                    suffix=f"ne-retrain-{plan.hypothesis_id}-{name}",
                )
                print(f"{name}: submitted {state[name]}", flush=True)
            self.storage.put_text(state_key, json.dumps(state, indent=2))
        jobs = {name: state[name] for name in plan.jobs}
        if plan.failed_job_id:
            jobs["failed"] = plan.failed_job_id
        if on_step is not None:
            on_step(
                {
                    "kind": "retrain",
                    "jobs": jobs,
                    "hyperparameters": plan.hyperparameters,
                    "cost_usd": plan.cost_usd,
                }
            )

        trained = {state[name] for name in plan.jobs}
        finished = wait_for_jobs(
            self.tf,
            trained,
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
        scores = {
            "base": self.score_base(plan.model, self.heldout_path, adapters["failed"]),
            "failed": self.score_adapter(adapters["failed"], self.heldout_path),
            "fixed": self.score_adapter(adapters["fixed"], self.heldout_path),
        }
        for name, score in scores.items():
            print(f"{name}: {score:.3f} on {self.heldout_path}", flush=True)
        rows = self.heldout_path.read_text(encoding="utf-8").splitlines()
        return {
            "kind": "comparison",
            "metric": self.metric,
            "heldout": str(self.heldout_path),
            "heldout_examples": sum(1 for r in rows if r.strip()),
            "heldout_note": self.heldout_note,
            "models": [
                {"name": "base", "label": plan.model, "score": scores["base"]},
                {
                    "name": "failed",
                    "label": "failed job",
                    "job_id": jobs["failed"],
                    "score": scores["failed"],
                },
                {
                    "name": "fixed",
                    "label": "retrained with the fix",
                    "job_id": jobs["fixed"],
                    "score": scores["fixed"],
                },
            ],
            "gain": round(scores["fixed"] - scores["failed"], 4),
            "cost_usd": plan.cost_usd,
        }
