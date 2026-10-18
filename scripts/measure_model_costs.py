"""Measure a bounded BANKING77 size/epoch grid; resume only known job IDs."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from bench.faults.recipe import HEALTHY_HP
from core.connectors.token_factory import TokenFactory
from scripts.tf_fault_sweep import summarize

MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "meta-llama/Llama-3.2-1B-Instruct"]
GRID = [(50, 1), (200, 1), (1000, 1), (200, 3), (500, 2)]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build_plan(rows, valid, budget):
    # Reserve at $1/M and allow 512 template tokens per example, plus one token
    # per serialized UTF-8 byte. This exceeds the recorded $0.40/M short-context rate.
    plans = []
    for model in MODELS:
        for n, epochs in GRID:
            text = "\n".join(rows[:n]) + "\n"
            upper = (len(text.encode()) + 512 * n) * epochs / 1e6
            plans.append(
                {
                    "run": f"{model.split('/')[-1]}-n{n}-e{epochs}",
                    "model": model,
                    "examples": n,
                    "epochs": epochs,
                    "reservation_usd": upper,
                    "train_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
    plan = {
        "runs": plans,
        "reserved_usd": sum(p["reservation_usd"] for p in plans),
        "validation_sha256": hashlib.sha256(valid).hexdigest(),
        "budget_usd": budget,
        "price_ceiling_per_million": 1.0,
    }
    if not 0 < budget <= 10 or plan["reserved_usd"] > budget:
        raise ValueError(f"Plan exceeds allowed budget: {plan['reserved_usd']}")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = Path("artifacts/b6-llama-8b-n1000/train.jsonl").read_text().splitlines()
    valid = Path("artifacts/b6-llama-8b-n1000/valid.jsonl").read_bytes()
    if len(rows) != 1000:
        raise ValueError("Expected exactly 1000 source training examples")
    args.out.mkdir(parents=True, exist_ok=True)
    plan = build_plan(rows, valid, args.budget)
    plan_path = args.out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Resume plan or source files changed")
    save(plan_path, plan)
    print(
        json.dumps({"jobs": len(plan["runs"]), "reserved_usd": plan["reserved_usd"]}),
        flush=True,
    )
    if not args.execute:
        return
    tf = TokenFactory.from_settings()
    tf.client = tf.client.with_options(timeout=60, max_retries=0)
    state_path = args.out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    valid_path = args.out / "valid.jsonl"
    valid_path.write_bytes(valid)
    if "_valid" not in state:
        state["_valid"] = tf.upload(valid_path)
        save(state_path, state)
    while True:
        active = []
        for p in plan["runs"]:
            name = p["run"]
            if (args.out / f"{name}.json").exists():
                continue
            if state.get(name) == "submission_pending":
                raise RuntimeError(
                    f"Unknown submission outcome for {name}; reconcile API before resuming"
                )
            if name in state:
                job = tf.job(state[name])
                if job.status in {"succeeded", "failed", "cancelled"}:
                    run = {
                        **p,
                        "hyperparameters": {**HEALTHY_HP, "n_epochs": p["epochs"]},
                    }
                    result = summarize(
                        job,
                        tf.events(job.id),
                        tf.checkpoints(job.id) if job.status == "succeeded" else [],
                        run,
                    )
                    save(args.out / f"{name}.json", result)
                    print(name, job.status, result["cost_usd"], flush=True)
                else:
                    active.append(name)
            elif len(active) < 3:
                path = args.out / f"train-n{p['examples']}.jsonl"
                path.write_text("\n".join(rows[: p["examples"]]) + "\n")
                key = f"_train{p['examples']}"
                if key not in state:
                    state[key] = tf.upload(path)
                state[name] = "submission_pending"
                save(state_path, state)
                job_id = tf.create_job(
                    p["model"],
                    state[key],
                    validation_file=state["_valid"],
                    hyperparameters={
                        **HEALTHY_HP,
                        "n_epochs": p["epochs"],
                        "lora": True,
                        "packing": False,
                    },
                    seed=0,
                    suffix="b6-matrix",
                )
                state[name] = job_id
                save(state_path, state)
                active.append(name)
                print(name, job_id, flush=True)
        if all((args.out / f"{p['run']}.json").exists() for p in plan["runs"]):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
