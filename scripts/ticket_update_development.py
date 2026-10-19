"""Frozen development pilot with inherited spending and fail-closed paid submission.

This is a new design. Passing only permits preparation of a separate confirmatory
protocol; it cannot revive the stopped historical run or open its final test.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess
from pathlib import Path

from bench.faults.data import truncate_answers
from bench.tasks.structured import sha
from bench.tasks.ticket_updates import build
from core.case import EvaluatorSpec
from core.connectors.storage import LocalStorage
from core.db.spending import SpendingLedger
from core.eval.data import read_examples
from core.eval.runtime import verify_runtime
from core.eval.structured import json_fields_exact
from core.interventions.tf_runner import download_last_checkpoint, weights_digest
from core.probes.level0.lengths import count_tokens
from scripts.fresh_answer_calibration import provider, save
from scripts.fresh_answer_power import decide as power_decide

SEEDS = (1201, 1202, 1203)
SOURCES = (
    "bench/tasks/ticket_updates.py",
    "scripts/ticket_update_development.py",
    "scripts/fresh_answer_calibration.py",
    "scripts/fresh_answer_power.py",
    "core/eval/__init__.py",
    "core/eval/data.py",
    "core/eval/generate.py",
    "core/eval/structured.py",
    "core/eval/runtime.py",
    "bench/causal_power.py",
    "core/interventions/stats.py",
    "core/db/spending.py",
    "uv.lock",
)


def prepare(root, old):
    from transformers import AutoTokenizer

    root.mkdir(parents=True, exist_ok=False)
    prior = json.loads((old / "protocol.json").read_text())
    completion = json.loads((old / "completion.json").read_text())
    if json.loads((old / "power-decision.json").read_text())["status"] != "stopped":
        raise ValueError("Previous run must be terminal")
    runtime = prior["evaluation_runtime"]
    verify_runtime(runtime)
    tokenizer = AutoTokenizer.from_pretrained(
        runtime["snapshot"], local_files_only=True
    )
    datasets = build(root / "data")
    clean = {
        r: [
            json.loads(line)
            for line in Path(datasets[r]["path"]).read_text().splitlines()
        ]
        for r in ("train", "validation")
    }
    damaged = {}
    injection = {}
    for i, (role, rows) in enumerate(clean.items()):
        damaged[role], injection[role] = truncate_answers(
            rows, 0.3, 2026101902 + i, 0.5, "random"
        )
    save(root / "evaluator-only-injection.json", injection)
    jobs = {}
    for arm, rows in (("healthy", clean), ("damaged", damaged)):
        valid = root / f"{arm}-validation.jsonl"
        valid.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows["validation"])
        )
        for seed in SEEDS:
            order = list(rows["train"])
            random.Random(seed).shuffle(order)
            path = root / f"{arm}-s{seed}-train.jsonl"
            path.write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in order)
            )
            tokens = [count_tokens(tokenizer, row["messages"]) for row in order]
            jobs[f"{arm}-s{seed}"] = {
                "train": str(path.resolve()),
                "train_sha256": sha(path),
                "validation": str(valid.resolve()),
                "validation_sha256": sha(valid),
                "examples": len(order),
                "seed": seed,
                "max_tokens": max(tokens),
                "rendered_tokens": sum(tokens),
                "reservation_usd": len(order) * max(tokens) * 0.4 / 1e6,
            }
    # SQLite's backup copies the full ledger consistently, including unsettled rows.
    source = sqlite3.connect(
        f"file:{(old / 'spending.sqlite3').resolve()}?mode=ro", uri=True
    )
    destination = sqlite3.connect(root / "spending.sqlite3")
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    snapshot = SpendingLedger(root / "spending.sqlite3", 5).snapshot()
    if any(r["status"] != "settled" for r in snapshot["operations"]):
        raise ValueError("Resolve inherited holds before new pilot")
    save(root / "inherited-spending.json", snapshot)
    estimate = sum(j["reservation_usd"] for j in jobs.values())
    if estimate + 1.5 > snapshot["available_usd"]:
        raise ValueError("Pilot plus contingency does not fit inherited cap")
    protocol = {
        "id": "ticket-updates-development-20261019-v1",
        "phase": "development_only",
        "model": prior["model"],
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {p: sha(Path(p)) for p in SOURCES},
        "datasets": datasets,
        "review_sha256": sha(root / "data/review.json"),
        "injection_sha256": sha(root / "evaluator-only-injection.json"),
        "evaluation_runtime": runtime,
        "evaluator": prior["evaluator"],
        "recipe": prior["recipe"],
        "jobs": jobs,
        "price": prior["price"],
        "cap_usd": 5,
        "pilot_reservation_usd": estimate,
        "budget": {
            "inherited_committed_usd": snapshot["committed_usd"],
            "contingency_usd": 0.5,
            "future_inference_reserve_usd": 1,
            "inherited_history_sha256": sha(root / "inherited-spending.json"),
            "source": str((old / "spending.sqlite3").resolve()),
            "policy": "This copied full-history ledger is the sole successor; never submit new work against the old ledger.",
        },
        "predecessor": {
            "protocol_sha256": sha(old / "protocol.json"),
            "completion": completion,
            "remains_terminal": True,
        },
        "base_gate": {"minimum": 0.10, "maximum": 0.90},
        "sensitivity_gate": {
            "healthy_minimum": 0.10,
            "healthy_maximum": 0.95,
            "minimum_healthy_gain_over_base": 0.05,
            "minimum_healthy_minus_damaged": 0.08,
        },
        "power_rule": {**prior["power_rule"], "random_seed": 3201},
        "stopping": "One base check; stop before payment outside [0.10,0.90]. Exactly three shuffled seeds per healthy/damaged arm, no adaptive repeats. Stop on weak learning/sensitivity, floor/ceiling, zero SD, duplicate weights, insufficient power, hash change, budget exhaustion or ambiguous provider state. A pass only permits a separate confirmatory protocol. No final test exists in this pilot.",
        "scope": "Synthetic ticket-state replay, shared grammar across development splits. Single task/model, no unique-cause claim, T=1, experimental classifier disabled.",
    }
    save(root / "protocol.json", protocol)
    (root / "protocol.sha256").write_text(sha(root / "protocol.json") + "\n")
    print(
        json.dumps(
            {
                "protocol_sha256": sha(root / "protocol.json"),
                "pilot_reservation_usd": estimate,
                "inherited_spend": snapshot["committed_usd"],
            }
        ),
        flush=True,
    )


def load(root):
    digest = (root / "protocol.sha256").read_text().strip()
    if sha(root / "protocol.json") != digest:
        raise ValueError("Protocol changed")
    plan = json.loads((root / "protocol.json").read_text())
    for path, expected in plan["source_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Source changed: {path}")
    for entry in plan["datasets"].values():
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Data changed")
    for path, expected in (
        (root / "data/review.json", plan["review_sha256"]),
        (root / "evaluator-only-injection.json", plan["injection_sha256"]),
        (root / "inherited-spending.json", plan["budget"]["inherited_history_sha256"]),
    ):
        if sha(path) != expected:
            raise ValueError("Review/injection/budget history changed")
    for job in plan["jobs"].values():
        for role in ("train", "validation"):
            if sha(Path(job[role])) != job[f"{role}_sha256"]:
                raise ValueError("Job data changed")
    verify_runtime(plan["evaluation_runtime"])
    state = (
        json.loads((root / "state.json").read_text())
        if (root / "state.json").exists()
        else {}
    )
    return plan, state, digest


def checked_score(path, plan, digest):
    record = json.loads(path.read_text())
    result = record["result"]
    if (
        record["protocol_sha256"] != digest
        or result["data_sha256"] != plan["datasets"]["check"]["sha256"]
        or result["runtime"] != plan["evaluation_runtime"]
        or result["metric"] != plan["evaluator"]["metric"]
    ):
        raise ValueError("Score binding changed")
    examples = read_examples(plan["datasets"]["check"]["path"])
    if len(examples) != len(result["examples"]):
        raise ValueError("Incomplete score")
    scores = []
    for expected, row in zip(examples, result["examples"], strict=True):
        value = json_fields_exact(row["answer"], expected.expected)
        if row["expected"] != expected.expected or row["score"] != value:
            raise ValueError("Saved prediction score changed")
        scores.append(value)
    if result["value"] != sum(scores) / len(scores):
        raise ValueError("Score mean changed")
    return record


def base_gate(value, rule):
    return rule["minimum"] <= value <= rule["maximum"]


def sensitivity(base, groups, rule):
    healthy = sum(groups["healthy"]) / len(groups["healthy"])
    damaged = sum(groups["damaged"]) / len(groups["damaged"])
    failures = []
    if not rule["healthy_minimum"] <= healthy <= rule["healthy_maximum"]:
        failures.append("healthy_floor_or_ceiling")
    if healthy - base < rule["minimum_healthy_gain_over_base"]:
        failures.append("insufficient_learning")
    if healthy - damaged < rule["minimum_healthy_minus_damaged"]:
        failures.append("insufficient_damage_sensitivity")
    return failures


def score(root, base=False):
    from core.eval import evaluate

    plan, state, digest = load(root)
    if (root / "decision.json").exists():
        raise ValueError("Pilot already stopped or completed")
    if not base and (
        set(state) != set(plan["jobs"])
        or any(e["status"] != "succeeded" for e in state.values())
    ):
        raise ValueError("Require all six successful jobs")
    entries = {"base": {}} if base else state
    for name, entry in entries.items():
        output = root / "scores" / f"{name}.json"
        if output.exists():
            continue
        if (
            not base
            and weights_digest(Path(entry["adapter"])) != entry["weights_sha256"]
        ):
            raise ValueError("Adapter changed")
        result = evaluate(
            plan["datasets"]["check"]["path"],
            EvaluatorSpec.model_validate(plan["evaluator"]),
            adapter=entry.get("adapter"),
            base_model=plan["model"],
            runtime=plan["evaluation_runtime"],
            device="mps",
        )
        save(
            output,
            {
                "protocol_sha256": digest,
                "job_id": entry.get("job_id"),
                "weights_sha256": entry.get("weights_sha256"),
                "result": result.to_dict(),
            },
        )
        print(name, result.value, flush=True)
    if base:
        value = json.loads((root / "scores/base.json").read_text())["result"]["value"]
        if not base_gate(value, plan["base_gate"]):
            save(
                root / "decision.json",
                {
                    "status": "stopped",
                    "reason": "base_floor_or_ceiling",
                    "base": value,
                    "protocol_sha256": digest,
                    "paid_jobs": 0,
                },
            )


def submit(root, env):
    plan, state, digest = load(root)
    if (root / "decision.json").exists():
        raise ValueError("Terminal pilot")
    base = checked_score(root / "scores/base.json", plan, digest)
    if base["protocol_sha256"] != digest or not base_gate(
        base["result"]["value"], plan["base_gate"]
    ):
        raise ValueError("Base gate not met")
    tf = provider(env)
    ledger = SpendingLedger(root / "spending.sqlite3", 5)
    for name, job in plan["jobs"].items():
        if name in state:
            if state[name].get("job_id"):
                continue
            raise ValueError("Ambiguous submission; reconcile provider before retrying")
        operation = f"{plan['id']}:{name}"
        ledger.reserve(
            operation, "ticket_updates_development_training", job["reservation_usd"]
        )
        state[name] = {
            "status": "submitting",
            "operation_id": operation,
            "protocol_sha256": digest,
        }
        save(root / "state.json", state)
        train_id = tf.upload(Path(job["train"]))
        valid_id = tf.upload(Path(job["validation"]))
        state[name].update(training_file=train_id, validation_file=valid_id)
        save(root / "state.json", state)
        job_id = tf.create_job(
            plan["model"],
            train_id,
            validation_file=valid_id,
            hyperparameters=plan["recipe"],
            seed=job["seed"],
            suffix=f"ne-update-dev-{name}",
        )
        state[name].update(job_id=job_id, status="submitted")
        save(root / "state.json", state)
        print(name, job_id, flush=True)


def collect(root, env):
    plan, state, _digest = load(root)
    tf = provider(env)
    ledger = SpendingLedger(root / "spending.sqlite3", 5)
    for name, entry in state.items():
        if not entry.get("job_id"):
            raise ValueError("Unresolved submission")
        job = tf.job(entry["job_id"])
        entry["status"] = job.status
        save(root / "provider" / f"{name}.json", job.model_dump())
        if job.status == "succeeded":
            if not entry.get("adapter"):
                adapter = download_last_checkpoint(
                    tf, LocalStorage(root), job.id, "checkpoints"
                )
                entry.update(
                    adapter=str(adapter.resolve()),
                    weights_sha256=weights_digest(adapter),
                )
            if not entry.get("settled"):
                if type(job.trained_tokens) is not int or job.trained_tokens < 0:
                    raise ValueError("Missing provider token count")
                cost = job.trained_tokens * plan["price"]["usd_per_million"] / 1e6
                ledger.settle(entry["operation_id"], cost)
                entry.update(
                    settled=True,
                    trained_tokens=job.trained_tokens,
                    estimated_cost_usd=cost,
                )
        save(root / "state.json", state)
        print(name, job.status, flush=True)
    save(root / "spending.json", ledger.snapshot())


def analyze(root):
    plan, state, digest = load(root)
    if (root / "decision.json").exists():
        raise ValueError("Terminal pilot; no repeated decision")
    groups = {"healthy": [], "damaged": []}
    digests = {"healthy": [], "damaged": []}
    if set(state) != set(plan["jobs"]):
        raise ValueError("Incomplete jobs")
    for name, entry in state.items():
        row = checked_score(root / "scores" / f"{name}.json", plan, digest)
        if (
            entry["status"] != "succeeded"
            or row["job_id"] != entry["job_id"]
            or row["protocol_sha256"] != digest
            or row["weights_sha256"] != weights_digest(Path(entry["adapter"]))
        ):
            raise ValueError("Score provenance mismatch")
        if (
            row["result"]["data_sha256"] != plan["datasets"]["check"]["sha256"]
            or row["result"]["runtime"] != plan["evaluation_runtime"]
        ):
            raise ValueError("Score data/runtime mismatch")
        arm = name.split("-", 1)[0]
        groups[arm].append(row["result"]["value"])
        digests[arm].append(row["weights_sha256"])
    base = json.loads((root / "scores/base.json").read_text())["result"]["value"]
    failures = sensitivity(base, groups, plan["sensitivity_gate"])
    result = (
        {
            "status": "stopped",
            "reason": ",".join(failures),
            "groups": groups,
            "base": base,
            "selected_seeds": [],
        }
        if failures
        else power_decide(groups, digests, plan["power_rule"])
    )
    result.update(
        protocol_sha256=digest,
        weight_digests=digests,
        next_phase="Separate confirmatory protocol required if passed; historical protocol remains stopped.",
    )
    save(root / "decision.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "base", "submit", "collect", "score", "analyze")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.old)
    elif args.action in ("base", "score"):
        score(args.root, base=args.action == "base")
    elif args.action in ("submit", "collect"):
        {"submit": submit, "collect": collect}[args.action](args.root, args.env_file)
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
