"""Run an intervention pair as Token Factory Post-training jobs.

Each arm trains once per seed as its own job, and all jobs go out at once. Token
Factory ignores the job's seed: two jobs that differ only in seed train identically.
So each seed gets its own order of the training file, shuffled here. For each
successful job the last checkpoint is downloaded into local storage and scored by
`evaluate`, which gets the adapter directory and returns the metric. The evaluator
is injected (jobs/eval_local/exact_match.py on a MacBook), so this module needs no
torch. Job IDs are saved under <prefix>/jobs.json as they are created, so a rerun
waits for the same jobs instead of paying for new ones.

Training on Token Factory is deterministic: the same file and hyperparameters give a
byte-identical adapter. If two seeds of an arm still train identical weights, the
seeds add no variance, and the runner says so.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TERMINAL_STATUSES, TokenFactory
from core.interventions.pairs import Intervention, take_fraction

# Token Factory needs these on top of the recipe for a LoRA job without packing
TF_LORA = {"lora": True, "packing": False}


@dataclass
class TokenFactoryPairRunner:
    tf: TokenFactory
    storage: LocalStorage
    model: str
    train_path: Path
    valid_path: Path
    base_hp: dict[str, Any]
    evaluate: Callable[[Path], float]
    prefix: str
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    poll_s: float = 30
    timeout_s: float = 3 * 3600
    sleep: Callable[[float], None] = time.sleep

    def __call__(self, target: Intervention, control: Intervention) -> None:
        state_key = f"{self.prefix}/jobs.json"
        state = (
            json.loads(self.storage.get_text(state_key))
            if self.storage.exists(state_key)
            else {}
        )
        valid_id = state.get("_valid") or self.tf.upload(self.valid_path)
        state["_valid"] = valid_id
        runs: list[tuple[Intervention, int, str]] = []
        for arm in (target, control):
            for seed in self.seeds:
                name = f"{arm.hypothesis_id}-{arm.kind}-s{seed}"
                train_key = f"_train_{arm.kind}_s{seed}"
                if train_key not in state:
                    state[train_key] = self.tf.upload(self._train_file(arm, seed))
                if name not in state:
                    state[name] = self.tf.create_job(
                        self.model,
                        state[train_key],
                        validation_file=valid_id,
                        hyperparameters={
                            **arm.hyperparameters(self.base_hp),
                            **TF_LORA,
                        },
                        seed=seed,
                        suffix=f"ne-{name}",
                    )
                    self.storage.put_text(state_key, json.dumps(state, indent=2))
                    print(f"{name}: submitted {state[name]}", flush=True)
                runs.append((arm, seed, state[name]))
        self.storage.put_text(state_key, json.dumps(state, indent=2))

        finished = wait_for_jobs(
            self.tf,
            {job_id for _, _, job_id in runs},
            poll_s=self.poll_s,
            timeout_s=self.timeout_s,
            sleep=self.sleep,
        )
        weights: dict[tuple[str, str], int] = {}  # (arm, sha256) -> first seed
        for arm, seed, job_id in runs:
            job = finished[job_id]
            if job.status != "succeeded":
                raise RuntimeError(f"{job_id} ended {job.status}: {job.error}")
            adapter_dir = download_last_checkpoint(
                self.tf, self.storage, job_id, self.prefix
            )
            digest = weights_digest(adapter_dir)
            if digest and (arm.kind, digest) in weights:
                print(
                    f"warning: {arm.kind} seeds {weights[arm.kind, digest]} and {seed} "
                    "trained identical weights, so they add no variance",
                    flush=True,
                )
            weights.setdefault((arm.kind, digest), seed)
            arm.metrics.append(self.evaluate(adapter_dir))
            arm.ft_job_ids.append(job_id)
            print(
                f"{arm.kind} seed {seed}: {arm.metrics[-1]:.3f} ({job_id})", flush=True
            )

    def _train_file(self, arm: Intervention, seed: int) -> Path:
        """The arm's share of the examples, in an order of its own for each seed."""
        rows = self.train_path.read_text(encoding="utf-8").splitlines()
        rows = take_fraction([r for r in rows if r.strip()], arm.data_fraction)
        random.Random(seed).shuffle(rows)
        key = f"{self.prefix}/{arm.kind}-s{seed}-train.jsonl"
        self.storage.put_text(key, "\n".join(rows) + "\n")
        return self.storage.root / key


def wait_for_jobs(
    tf: TokenFactory,
    job_ids: set[str],
    *,
    poll_s: float = 30,
    timeout_s: float = 3 * 3600,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll the jobs until each has ended; prints every change of status or steps."""
    deadline = time.monotonic() + timeout_s
    finished: dict[str, Any] = {}
    seen: dict[str, tuple] = {}
    while len(finished) < len(job_ids):
        for job_id in sorted(job_ids - set(finished)):
            job = tf.job(job_id)
            now = (job.status, job.trained_steps, job.total_steps)
            if seen.get(job_id) != now:
                print(f"{job_id}: {job.status} steps {now[1]}/{now[2]}", flush=True)
                seen[job_id] = now
            if job.status in TERMINAL_STATUSES:
                finished[job_id] = job
        if len(finished) < len(job_ids):
            if time.monotonic() > deadline:
                raise TimeoutError(f"jobs still running: {job_ids - set(finished)}")
            sleep(poll_s)
    return finished


def weights_digest(adapter_dir: Path) -> str:
    """SHA-256 of the adapter weights, or "" when the checkpoint has none."""
    path = adapter_dir / "adapter_model.safetensors"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def download_last_checkpoint(
    tf: TokenFactory, storage: LocalStorage, job_id: str, prefix: str
) -> Path:
    """The job's last checkpoint in local storage; returns its directory."""
    last = max(tf.checkpoints(job_id), key=lambda c: c.step_number)
    keys = tf.download_checkpoint(last, storage, prefix)
    return storage.root / Path(keys[0]).parent
