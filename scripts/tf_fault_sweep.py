"""B9 on Token Factory: a grid of real LoRA jobs, healthy and with faults 6–8.

Uploads the train and validation files once and keeps up to --parallel jobs running.
For each finished job it writes <out>/<run>.json with the hyperparameters, per-epoch
train_loss and valid_loss from the checkpoints, trained tokens, cost and seconds from
job creation to each stage. Job IDs go to <out>/jobs.json as they are submitted, so a
rerun with the same arguments resumes and never submits a run twice.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError

from bench.faults.recipe import HEALTHY_HP, STRENGTHS, grid, inject
from core.connectors.token_factory import (
    TERMINAL_STATUSES,
    TokenFactory,
    hyperparameter_problem,
)
from core.planner.job_cost import price_per_m

# Token Factory needs these on top of the shared recipe
TF_ONLY_HP = {"lora": True, "packing": False}


SHORT = {"lr_too_low": "lr", "too_many_epochs": "ep", "lora_rank_too_small": "rank"}


def plan_runs(
    model: str, healthy: dict[str, Any], seeds: list[int], faults: list[str] | None
) -> list[dict[str, Any]]:
    """Runs of the grid; one that Token Factory would reject is printed and left out."""
    runs = []
    for name, fault, strength, seed in grid(seeds, faults):
        hp = inject(healthy, fault, strength) if fault else dict(healthy)
        why = hyperparameter_problem(hp)
        if why:
            print(f"{name}-s{seed}: {why}, skipped", flush=True)
            continue
        runs.append(
            {
                "run": f"{name}-s{seed}",
                "fault": fault,
                "strength": strength,
                "seed": seed,
                "model": model,
                "hyperparameters": hp,
            }
        )
    return runs


def suffix(run: dict[str, Any]) -> str:
    """Output model suffix such as b9-lr-mild-s0."""
    name = run["run"]
    for long, short in SHORT.items():
        name = name.replace(long, short)
    return f"b9-{name}"


def stage_seconds(job: Any, events: list[Any]) -> dict[str, int]:
    """Seconds from job creation to the datasets, training start and end, finish."""
    at: dict[str, int] = {}
    for e in events:
        if e.message.startswith("Dataset"):
            at["datasets"] = max(at.get("datasets", 0), e.created_at)
        elif e.message == "Training started":
            at["training_started"] = e.created_at
        elif e.message.startswith("Training completed"):
            at["training_completed"] = e.created_at
    if job.finished_at:
        at["finished"] = job.finished_at
    return {k: v - job.created_at for k, v in at.items()}


def summarize(
    job: Any, events: list[Any], checkpoints: list[Any], run: dict[str, Any]
) -> dict[str, Any]:
    cks = sorted(checkpoints, key=lambda c: c.step_number)
    tokens = job.trained_tokens or 0
    try:
        price = price_per_m(run["model"], run["hyperparameters"]["context_length"])
    except KeyError:
        price = None
    return {
        **run,
        "job_id": job.id,
        "status": job.status,
        "error": str(job.error) if job.error else None,
        "trained_tokens": tokens,
        "cost_usd": tokens * price / 1e6 if price is not None else None,
        "steps": [c.step_number for c in cks],
        "train_loss": [c.metrics.train_loss for c in cks],
        "valid_loss": [c.metrics.valid_loss for c in cks],
        "seconds": stage_seconds(job, events),
    }


def run_grid(
    tf: Any,
    runs: list[dict[str, Any]],
    out: Path,
    train_path: str,
    valid_path: str,
    parallel: int = 3,
    poll_s: float = 30,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "jobs.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def save_state() -> None:
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    pending = [r for r in runs if not (out / f"{r['run']}.json").exists()]
    if pending and "_files" not in state:
        state["_files"] = {
            "train": tf.upload(train_path),
            "valid": tf.upload(valid_path),
        }
        save_state()
    by_name = {r["run"]: r for r in runs}
    active = {r["run"]: state[r["run"]] for r in pending if r["run"] in state}
    queue = [r for r in pending if r["run"] not in state]
    seen: dict[str, tuple] = {}

    while queue or active:
        while queue and len(active) < parallel:
            run = queue.pop(0)
            job_id = tf.create_job(
                run["model"],
                state["_files"]["train"],
                validation_file=state["_files"]["valid"],
                hyperparameters={**run["hyperparameters"], **TF_ONLY_HP},
                seed=run["seed"],
                suffix=suffix(run),
            )
            state[run["run"]] = active[run["run"]] = job_id
            save_state()
            print(f"{run['run']}: submitted {job_id}", flush=True)
        sleep(poll_s)
        for name, job_id in list(active.items()):
            try:
                job = tf.job(job_id)
                if job.status not in TERMINAL_STATUSES:
                    now = (job.status, job.trained_steps, job.total_steps)
                    if seen.get(name) != now:
                        print(f"{name}: steps {now[1]}/{now[2]}", flush=True)
                        seen[name] = now
                    continue
                result = summarize(
                    job, tf.events(job_id), tf.checkpoints(job_id), by_name[name]
                )
            except (APIConnectionError, APIStatusError) as exc:
                print(f"{name}: request failed ({type(exc).__name__})", flush=True)
                continue
            path = out / f"{name}.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            del active[name]
            print(
                f"{name}: {result['status']}, valid_loss {result['valid_loss']}",
                flush=True,
            )

    results = [
        json.loads((out / f"{r['run']}.json").read_text())
        for r in runs
        if (out / f"{r['run']}.json").exists()
    ]
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hp", default="{}", help="JSON overrides of the healthy run")
    parser.add_argument(
        "--faults", nargs="*", choices=sorted(STRENGTHS), help="default: all"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--poll", type=float, default=30)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the runs, submit nothing"
    )
    args = parser.parse_args()

    healthy = {**HEALTHY_HP, **json.loads(args.hp)}
    runs = plan_runs(args.model, healthy, args.seeds, args.faults)
    if args.dry_run:
        for r in runs:
            print(r["run"], suffix(r), json.dumps(r["hyperparameters"]))
        return
    results = run_grid(
        TokenFactory.from_settings(),
        runs,
        Path(args.out),
        args.train,
        args.valid,
        parallel=args.parallel,
        poll_s=args.poll,
    )
    cost = sum(r["cost_usd"] or 0 for r in results)
    print(f"{len(results)} runs in {args.out}/results.jsonl, about ${cost:.3f}")


if __name__ == "__main__":
    main()
