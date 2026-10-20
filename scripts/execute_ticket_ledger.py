"""Execute the frozen ticket-ledger admission study within the additional budget."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import random
from pathlib import Path

from bench.tasks.ticket_ledger import ARMS, FAMILIES
from core.connectors.storage import LocalStorage
from core.db.spending import SpendingLedger
from core.eval.data import read_examples
from core.eval.format_content import evaluate_example
from core.eval.generate import free_cache, load_model
from core.eval.runtime import file_digest, freeze_runtime, verify_runtime
from core.eval.structured import json_fields_exact, validate_answer
from core.interventions.tf_runner import download_last_checkpoint, weights_digest
from core.probes.level0.lengths import count_tokens
from scripts.fresh_answer_calibration import provider, save
from scripts.prepare_ticket_ledger import verify

AUTHORIZATION = Path(
    "docs/results/remaining-queue-authorization-20261020/authorization.json"
)
BUNDLE = Path("docs/results/ticket-ledger-development-20261020-v2")
SOURCES = (
    "scripts/execute_ticket_ledger.py",
    "scripts/fresh_answer_calibration.py",
    "core/connectors/token_factory.py",
    "core/db/spending.py",
    "core/interventions/tf_runner.py",
    "core/eval/data.py",
    "core/eval/records.py",
    "core/eval/generate.py",
    "core/eval/runtime.py",
    "core/eval/format_content.py",
    "core/eval/structured.py",
    "core/probes/level0/lengths.py",
)


def metrics(row):
    expected = validate_answer("json_extraction", row["expected"])
    try:
        value = validate_answer("json_extraction", row["answer"])
        valid = 1.0
        fields = sum(value[k] == expected[k] for k in expected) / len(expected)
    except (ValueError, TypeError, RecursionError):
        valid, fields = 0.0, 0.0
    return {
        "strict_exact": 0.0
        if row["hit_limit"]
        else json_fields_exact(row["answer"], row["expected"]),
        "schema_valid": valid,
        "field_accuracy": fields,
    }


def summaries(rows):
    result = {}
    for group in ("serialization", "history", *FAMILIES):
        selected = [
            row
            for row in rows
            if (
                row["source"] != "serialization"
                if group == "history"
                else row["source"] == group
            )
        ]
        if not selected:
            raise ValueError(f"Missing group {group}")
        result[group] = {
            key: sum(row[key] for row in selected) / len(selected)
            for key in (
                "strict_exact",
                "schema_valid",
                "field_accuracy",
                "hit_limit",
                "stopped_on_eos",
            )
        }
        result[group]["examples"] = len(selected)
    return result


def prepare(root, snapshot):
    from transformers import AutoTokenizer

    verify(BUNDLE)
    authorization = json.loads(AUTHORIZATION.read_text())
    if authorization["additional_cap_usd"] != 25:
        raise ValueError("Unexpected budget authorization")
    frozen = json.loads((BUNDLE / "protocol.json").read_text())
    runtime = freeze_runtime(snapshot, frozen["model"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    root.mkdir(parents=True, exist_ok=False)
    recipe = dict(frozen["recipe"])
    recipe["lora_r"] = recipe.pop("lora_rank")
    # Complete the previously unspecified provider defaults prospectively.
    recipe.update(warmup_ratio=0.0, weight_decay=0.0, max_grad_norm=1.0)
    rate = authorization["training_price_observation"][
        "usd_per_million_tokens_excluding_tax"
    ]
    multiplier = authorization["budget_charge_multiplier"]
    jobs = {}
    for arm in ARMS:
        rows = (BUNDLE / f"{arm}-train.jsonl").read_text().splitlines()
        for seed in frozen["order_seeds"]:
            ordered = list(rows)
            random.Random(seed).shuffle(ordered)
            name = f"{arm}-s{seed}"
            path = root / f"{name}.jsonl"
            path.write_text("\n".join(ordered) + "\n")
            tokens = [
                count_tokens(tokenizer, json.loads(row)["messages"]) for row in ordered
            ]
            if max(tokens) > recipe["context_length"]:
                raise ValueError("Context overflow")
            jobs[name] = {
                "arm": arm,
                "seed": seed,
                "train": str(path.resolve()),
                "train_sha256": file_digest(path),
                "rendered_tokens": sum(tokens),
                "rows": len(rows),
                "max_tokens": max(tokens),
                "reservation_usd": len(rows)
                * max(tokens)
                * recipe["n_epochs"]
                * rate
                / 1e6
                * multiplier,
                "token_cost_estimate_usd": sum(tokens)
                * recipe["n_epochs"]
                * rate
                / 1e6,
            }
    plan = {
        "id": "ticket-ledger-execution-20261020-v1",
        "authorization_sha256": file_digest(AUTHORIZATION),
        "data_protocol_sha256": file_digest(BUNDLE / "protocol.json"),
        "bundle_sha256": {str(p): file_digest(p) for p in sorted(BUNDLE.iterdir())},
        "source_sha256": {p: file_digest(p) for p in SOURCES},
        "evaluation_runtime": runtime,
        "model": frozen["model"],
        "recipe": recipe,
        "generation": frozen["generation"],
        "gates": frozen["gates"],
        "order_seeds": frozen["order_seeds"],
        "jobs": jobs,
        "validation": str((BUNDLE / "validation.jsonl").resolve()),
        "check": str((BUNDLE / "check.jsonl").resolve()),
        "spending_ledger": str(Path(authorization["ledger"]).resolve()),
        "cap_usd": 25,
        "training_rate_usd_per_million": rate,
        "budget_charge_multiplier": multiplier,
        "stopping": "No retries or extra seeds. Ambiguous/failed/cancelled jobs stop admission with holds retained. All gates required; a fail stops this design. No final test.",
        "evaluation": "One model at a time on MPS; greedy thinking off 192 tokens. Persist each row; resume only unscored rows. Existing complete scores are verified, not regenerated.",
    }
    ledger = SpendingLedger(plan["spending_ledger"], 25)
    if (
        sum(j["reservation_usd"] for j in jobs.values())
        > ledger.snapshot()["available_usd"]
    ):
        raise ValueError("Series does not fit approved additional budget")
    save(root / "protocol.json", plan)
    (root / "protocol.sha256").write_text(file_digest(root / "protocol.json") + "\n")
    print(
        json.dumps(
            {
                "jobs": len(jobs),
                "reservation_usd": sum(j["reservation_usd"] for j in jobs.values()),
            }
        )
    )


def load(root, *, runtime=False):
    digest = file_digest(root / "protocol.json")
    if digest != (root / "protocol.sha256").read_text().strip():
        raise ValueError("Protocol drift")
    plan = json.loads((root / "protocol.json").read_text())
    if file_digest(AUTHORIZATION) != plan["authorization_sha256"]:
        raise ValueError("Authorization drift")
    for path, expected in {**plan["bundle_sha256"], **plan["source_sha256"]}.items():
        if file_digest(path) != expected:
            raise ValueError(f"Source/data drift: {path}")
    for job in plan["jobs"].values():
        if file_digest(job["train"]) != job["train_sha256"]:
            raise ValueError("Training order drift")
    if runtime:
        verify_runtime(plan["evaluation_runtime"])
    state = (
        json.loads((root / "state.json").read_text())
        if (root / "state.json").exists()
        else {}
    )
    return plan, state, digest


def submit(root, env):
    plan, state, digest = load(root)
    if (root / "decision.json").exists():
        raise ValueError("Terminal study")
    ledger = SpendingLedger(plan["spending_ledger"], 25)
    remaining = sum(
        job["reservation_usd"]
        for name, job in plan["jobs"].items()
        if name not in state
    )
    if remaining > ledger.snapshot()["available_usd"]:
        raise ValueError("Insufficient uncommitted budget")
    tf = provider(env)
    for name, job in plan["jobs"].items():
        if name in state:
            if not state[name].get("job_id"):
                raise ValueError("Ambiguous attempt; reconcile without retry")
            if state[name]["status"] in {"failed", "cancelled"}:
                raise ValueError("Failed study; no further submissions")
            continue
        operation = f"{plan['id']}:{name}"
        ledger.reserve(operation, "ticket_ledger_admission", job["reservation_usd"])
        state[name] = {
            "operation_id": operation,
            "status": "submitting",
            "protocol_sha256": digest,
        }
        save(root / "state.json", state)
        entry = state[name]
        entry["training_file"] = tf.upload(Path(job["train"]))
        save(root / "state.json", state)
        entry["validation_file"] = tf.upload(Path(plan["validation"]))
        save(root / "state.json", state)
        entry["job_id"] = tf.create_job(
            plan["model"],
            entry["training_file"],
            validation_file=entry["validation_file"],
            hyperparameters=plan["recipe"],
            seed=job["seed"],
            suffix=f"ne-ledger-{name}",
        )
        entry["status"] = "submitted"
        save(root / "state.json", state)
        save(root / "spending.json", ledger.snapshot())
        print(name, entry["job_id"], flush=True)


def collect(root, env):
    plan, state, _ = load(root)
    ledger = SpendingLedger(plan["spending_ledger"], 25)
    tf = provider(env)
    for name, entry in state.items():
        if not entry.get("job_id"):
            raise ValueError("Ambiguous attempt; reconcile without retry")
        job = tf.job(entry["job_id"])
        save(root / "provider" / f"{name}.json", job.model_dump())
        entry["status"] = job.status
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
                    raise ValueError("Missing trained-token count")
                estimate = (
                    job.trained_tokens * plan["training_rate_usd_per_million"] / 1e6
                )
                charge = estimate * plan["budget_charge_multiplier"]
                operation = next(
                    o
                    for o in ledger.snapshot()["operations"]
                    if o["id"] == entry["operation_id"]
                )
                if operation["status"] == "pending":
                    ledger.settle(entry["operation_id"], charge)
                elif abs(operation["actual_usd"] - charge) > 1e-12:
                    raise ValueError("Conflicting budget settlement")
                entry.update(
                    settled=True,
                    trained_tokens=job.trained_tokens,
                    token_cost_estimate_usd=estimate,
                    budget_charge_usd=charge,
                )
        save(root / "state.json", state)
        print(name, entry["status"], flush=True)
    save(root / "spending.json", ledger.snapshot())


def checked_rows(record, examples, binding):
    if any(record.get(k) != v for k, v in binding.items()):
        raise ValueError("Score binding differs")
    rows = record["rows"]
    if len(rows) > len(examples):
        raise ValueError("Too many scored rows")
    for row, example in zip(rows, examples):
        if (row["line"], row["expected"], row["source"]) != (
            example.line,
            example.expected,
            example.source,
        ):
            raise ValueError("Scored row differs from data")
        if any(row[k] != v for k, v in metrics(row).items()):
            raise ValueError("Metric drift")
    if record.get("complete") and (
        len(rows) != len(examples) or record["summary"] != summaries(rows)
    ):
        raise ValueError("Incomplete or altered summary")
    return rows


def score(root, base=False):
    plan, state, digest = load(root, runtime=True)
    if (root / "decision.json").exists():
        raise ValueError("Terminal study")
    if not base and (
        set(state) != set(plan["jobs"])
        or any(e["status"] != "succeeded" for e in state.values())
    ):
        raise ValueError("All planned jobs must succeed before adapter scoring")
    examples = read_examples(plan["check"])
    for name, entry in ({"base": {}} if base else state).items():
        binding = {
            "protocol_sha256": digest,
            "data_sha256": file_digest(plan["check"]),
            "runtime": plan["evaluation_runtime"],
            "job_id": entry.get("job_id"),
            "weights_sha256": entry.get("weights_sha256"),
        }
        path = root / "scores" / f"{name}.json"
        record = (
            json.loads(path.read_text())
            if path.exists()
            else {**binding, "rows": [], "complete": False}
        )
        rows = checked_rows(record, examples, binding)
        if record["complete"]:
            continue
        if (
            entry.get("adapter")
            and weights_digest(Path(entry["adapter"])) != entry["weights_sha256"]
        ):
            raise ValueError("Adapter drift")
        verify_runtime(plan["evaluation_runtime"], template_dir=entry.get("adapter"))
        model, tokenizer, _ = load_model(
            entry.get("adapter"), base_model=plan["evaluation_runtime"]["snapshot"]
        )
        model.to("mps").eval()
        for example in examples[len(rows) :]:
            row = evaluate_example(model, tokenizer, example, "mps")
            row.update(metrics(row))
            rows.append(row)
            save(path, record)
            if len(rows) % 12 == 0:
                print(name, len(rows), "/", len(examples), flush=True)
            free_cache("mps")
        record.update(complete=True, summary=summaries(rows))
        save(path, record)
        checked_rows(record, examples, binding)
        print(name, record["summary"], flush=True)
        del model, tokenizer
        gc.collect()
        free_cache("mps")


def gate_failures(scores, weights, plan):
    seeds = plan["order_seeds"]
    expected = {f"{arm}-s{seed}" for arm in ARMS for seed in seeds}
    if set(scores) != expected or set(weights) != expected:
        raise ValueError("All arm/seed results required")
    means = {
        arm: {
            group: sum(
                scores[f"{arm}-s{seed}"][group]["strict_exact"] for seed in seeds
            )
            / len(seeds)
            for group in ("serialization", "history", *FAMILIES)
        }
        for arm in ARMS
    }
    g = plan["gates"]
    failures = []
    for arm in ("format", "healthy"):
        if means[arm]["serialization"] < g["format_and_healthy_anchor_exact_min"]:
            failures.append(f"{arm}_anchor_floor")
    healthy = means["healthy"]["history"]
    if not g["healthy_history_exact_min"] <= healthy <= g["healthy_history_exact_max"]:
        failures.append("healthy_history_range")
    for arm in ("format", "wrong"):
        if (
            healthy - means[arm]["history"]
            < g[f"healthy_minus_{arm}_history_exact_min"]
        ):
            failures.append(f"healthy_minus_{arm}")
    for family in FAMILIES:
        if means["healthy"][family] < g["healthy_each_family_exact_min"]:
            failures.append(f"healthy_{family}_floor")
    for arm in ARMS:
        if (
            len({weights[f"{arm}-s{seed}"] for seed in seeds})
            != g["distinct_weight_digests_per_arm"]
        ):
            failures.append(f"{arm}_distinct_weights")
    return failures, means


def analyze(root):
    plan, state, digest = load(root)
    if set(state) != set(plan["jobs"]) or any(
        e["status"] != "succeeded" for e in state.values()
    ):
        raise ValueError("Incomplete training")
    scores, weights = {}, {}
    examples = read_examples(plan["check"])
    for name, entry in state.items():
        record = json.loads((root / "scores" / f"{name}.json").read_text())
        checked_rows(
            record,
            examples,
            {
                "protocol_sha256": digest,
                "data_sha256": file_digest(plan["check"]),
                "runtime": plan["evaluation_runtime"],
                "job_id": entry["job_id"],
                "weights_sha256": entry["weights_sha256"],
            },
        )
        if not record["complete"]:
            raise ValueError("Incomplete scoring")
        scores[name], weights[name] = record["summary"], entry["weights_sha256"]
    failures, means = gate_failures(scores, weights, plan)
    result = {
        "protocol_sha256": digest,
        "status": "stopped" if failures else "admitted_for_power_planning",
        "failed_gates": failures,
        "means": means,
        "spending": SpendingLedger(plan["spending_ledger"], 25).snapshot(),
        "scope": "Development admission only; no final test or causal confirmation.",
    }
    if (root / "decision.json").exists():
        raise FileExistsError("Decision already saved")
    save(root / "decision.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "submit", "collect", "score", "base", "analyze")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    args.root.parent.mkdir(parents=True, exist_ok=True)
    with (args.root.parent / "ticket-ledger.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "prepare":
            prepare(args.root, args.snapshot)
        elif args.action in {"submit", "collect"}:
            (submit if args.action == "submit" else collect)(args.root, args.env_file)
        elif args.action in {"score", "base"}:
            score(args.root, args.action == "base")
        else:
            analyze(args.root)


if __name__ == "__main__":
    main()
