"""Frozen development-only TF repeats for the fresh answer-cut protocol.

prepare is tokenizer-only; submit is paid and uses an explicit protocol digest.
collect downloads checkpoints without loading models. score requires a recorded
exclusive MPS handoff and never reads the fresh cohort or its final test.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

from bench.faults.data import truncate_answers
from bench.faults.recipe import HEALTHY_HP
from bench.tasks.structured import sha
from core.case import EvaluatorSpec
from core.config import Settings
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TokenFactory
from core.db.spending import SpendingLedger
from core.eval.runtime import file_digest, freeze_runtime, verify_runtime
from core.interventions.tf_runner import download_last_checkpoint, weights_digest
from core.probes.level0.lengths import count_tokens

MODEL = "Qwen/Qwen3-0.6B"
SEEDS = [1101, 1102, 1103]
SOURCE_FILES = [
    "scripts/fresh_answer_calibration.py",
    "core/eval/__init__.py",
    "core/eval/generate.py",
    "core/eval/runtime.py",
    "core/eval/structured.py",
    "core/interventions/stats.py",
    "bench/causal_power.py",
    "uv.lock",
]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare(root, snapshot):
    from transformers import AutoTokenizer

    root.mkdir(parents=True, exist_ok=False)
    runtime = freeze_runtime(snapshot, MODEL)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    old = Path("datasets/structured-mixed-v1/json_extraction")
    source = {
        role: {
            "path": str((old / f"{role}.jsonl").resolve()),
            "sha256": sha(old / f"{role}.jsonl"),
        }
        for role in ("train", "validation", "check", "test")
    }
    rows = {
        role: [
            json.loads(line)
            for line in Path(source[role]["path"]).read_text().splitlines()
        ]
        for role in ("train", "validation")
    }
    damaged = {
        role: truncate_answers(values, 0.3, 2026101901 + i, 0.5, "random")[0]
        for i, (role, values) in enumerate(rows.items())
    }
    jobs = {}
    for name, inputs in (("healthy", rows), ("damaged", damaged)):
        valid = root / f"{name}-validation.jsonl"
        valid.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in inputs["validation"]
            )
        )
        for seed in SEEDS:
            train = list(inputs["train"])
            random.Random(seed).shuffle(train)
            path = root / f"{name}-s{seed}-train.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in train)
            )
            tokens = [count_tokens(tokenizer, row["messages"]) for row in train]
            jobs[f"{name}-s{seed}"] = {
                "train": str(path.resolve()),
                "validation": str(valid.resolve()),
                "train_sha256": sha(path),
                "validation_sha256": sha(valid),
                "seed": seed,
                "examples": len(train),
                "rendered_tokens": sum(tokens),
                "max_tokens": max(tokens),
                "reservation_usd": len(train) * max(tokens) * 0.4 / 1e6,
            }
    plan = {
        "version": 1,
        "id": "fresh-answer-development-20261019",
        "phase": "development_only",
        "model": MODEL,
        "evaluation_runtime": runtime,
        "source_datasets": source,
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {path: sha(Path(path)) for path in SOURCE_FILES},
        "recipe": {**HEALTHY_HP, "lora": True, "packing": False},
        "jobs": jobs,
        "cap_usd": 5.0,
        "development_reservation_usd": sum(j["reservation_usd"] for j in jobs.values()),
        "price": {
            "usd_per_million": 0.4,
            "product": "Fine-tuning Qwen3-0.6B LoRa 8K",
            "source": "https://tokenfactory.nebius.com/organization/prices",
            "verified_at": "2026-10-19",
            "valid_from": "2025-12-01",
            "tax_scope": "excluding taxes; contingency within $5 overall cap",
            "method": "authenticated rendered price table read before submission",
        },
        "provider_revision": "Provider does not expose immutable base weight revision in the documented job request; retain model ID, returned metadata and adapter hashes. Do not claim equality to local base weights.",
        "evaluator": {
            "name": "ticket_json_v1",
            "metric": "json_fields_exact",
            "parameters": {"max_new_tokens": 192},
        },
        "scoring_split": "historical check only; no fresh data",
        "stopping": "Exactly six jobs. No retry after ambiguous submission, no extra repeats after metrics. Stop on source/runtime/data change or budget exhaustion.",
        "power_rule": {
            "min_effect": 0.08,
            "anticipated_effect": 0.16,
            "candidate_seeds": [3, 4, 5, 6, 7, 8],
            "trials": 1000,
            "random_seed": 3101,
            "minimum_lower_mc_bound": 0.8,
            "zero_sd": "stop and report; do not import a noise floor",
        },
    }
    save(root / "protocol.json", plan)
    digest = sha(root / "protocol.json")
    (root / "protocol.sha256").write_text(digest + "\n")
    print(
        json.dumps(
            {
                "protocol_sha256": digest,
                "development_reservation_usd": plan["development_reservation_usd"],
                "jobs": len(jobs),
            }
        )
    )


def load(root, expected):
    if sha(root / "protocol.json") != expected:
        raise ValueError("Protocol digest changed")
    plan = json.loads((root / "protocol.json").read_text())
    for path, digest in plan["source_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    for entry in plan["source_datasets"].values():
        if file_digest(entry["path"]) != entry["sha256"]:
            raise ValueError("Development source data changed")
    for job in plan["jobs"].values():
        for role in ("train", "validation"):
            if file_digest(job[role]) != job[f"{role}_sha256"]:
                raise ValueError("Prepared job data changed")
    verify_runtime(plan["evaluation_runtime"], model=plan["model"])
    state_path = root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    return plan, state


def provider(env_file):
    settings = Settings(_env_file=env_file)
    tf = TokenFactory.from_settings(settings)
    tf.client = tf.client.with_options(max_retries=0, timeout=120)
    return tf


def submit(root, expected, env_file):
    plan, state = load(root, expected)
    tf = provider(env_file)
    ledger = SpendingLedger(root / "spending.sqlite3", plan["cap_usd"])
    for name, job in plan["jobs"].items():
        if name in state:
            if state[name].get("job_id"):
                print(name, "already submitted", state[name]["job_id"], flush=True)
                continue
            raise RuntimeError(
                "Ambiguous attempt: reconcile provider state before retrying"
            )
        ledger.reserve(name, "development_training", job["reservation_usd"])
        state[name] = {"status": "submitting", "protocol_sha256": expected}
        save(root / "state.json", state)
        train_id, valid_id = (
            tf.upload(Path(job["train"])),
            tf.upload(Path(job["validation"])),
        )
        state[name].update(training_file=train_id, validation_file=valid_id)
        save(root / "state.json", state)
        job_id = tf.create_job(
            plan["model"],
            train_id,
            validation_file=valid_id,
            hyperparameters=plan["recipe"],
            seed=job["seed"],
            suffix=f"ne-fresh-dev-{name}",
        )
        state[name].update(job_id=job_id, status="submitted")
        save(root / "state.json", state)
        print(name, job_id, flush=True)
    print(json.dumps(ledger.snapshot()))


def collect(root, expected, env_file):
    plan, state = load(root, expected)
    tf = provider(env_file)
    storage = LocalStorage(root)
    ledger = SpendingLedger(root / "spending.sqlite3", plan["cap_usd"])
    for name, entry in state.items():
        if not entry.get("job_id"):
            raise RuntimeError("Unresolved submission")
        job = tf.job(entry["job_id"])
        entry["status"] = job.status
        save(root / "provider" / f"{name}.json", job.model_dump())
        if job.status == "succeeded":
            if not entry.get("adapter"):
                adapter = download_last_checkpoint(tf, storage, job.id, "checkpoints")
                entry["adapter"] = str(adapter.resolve())
                entry["weights_sha256"] = weights_digest(adapter)
            if not entry.get("settled"):
                if type(job.trained_tokens) is not int or job.trained_tokens < 0:
                    raise ValueError("Provider trained-token count unavailable")
                cost = job.trained_tokens * plan["price"]["usd_per_million"] / 1e6
                ledger.settle(name, cost)
                entry.update(
                    settled=True,
                    trained_tokens=job.trained_tokens,
                    token_cost_estimate_usd=cost,
                )
        save(root / "state.json", state)
        print(name, job.status, entry.get("trained_tokens"), flush=True)
    save(root / "spending.json", ledger.snapshot())


def score(root, expected):
    from core.eval import evaluate

    plan, state = load(root, expected)
    handoff = root / "mps-handoff.json"
    if (
        not handoff.exists()
        or json.loads(handoff.read_text()).get("released") is not True
    ):
        raise ValueError("Record the B9 exclusive resource handoff before scoring")
    if set(state) != set(plan["jobs"]) or any(
        s["status"] != "succeeded" for s in state.values()
    ):
        raise ValueError("All six development jobs must succeed before scoring")
    for name, entry in state.items():
        path = root / "scores" / f"{name}.json"
        if path.exists():
            continue
        if weights_digest(Path(entry["adapter"])) != entry["weights_sha256"]:
            raise ValueError("Adapter changed")
        result = evaluate(
            plan["source_datasets"]["check"]["path"],
            EvaluatorSpec.model_validate(plan["evaluator"]),
            adapter=entry["adapter"],
            base_model=plan["model"],
            runtime=plan["evaluation_runtime"],
            device="mps",
        )
        save(
            path,
            {
                "job_id": entry["job_id"],
                "protocol_sha256": expected,
                "weights_sha256": entry["weights_sha256"],
                "data_sha256": plan["source_datasets"]["check"]["sha256"],
                "result": result.to_dict(),
            },
        )
        print(name, result.value, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "collect", "score"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.snapshot)
    elif args.action == "score":
        score(args.root, args.protocol_sha256)
    else:
        {"submit": submit, "collect": collect}[args.action](
            args.root, args.protocol_sha256, args.env_file
        )


if __name__ == "__main__":
    main()
