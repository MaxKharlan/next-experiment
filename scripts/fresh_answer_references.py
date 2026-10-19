"""Three budgeted fresh references, allowed only after the development power gate.

No model evaluation or final-test access is performed by this command. Every
submission shares the development ledger and refuses an ambiguous retry.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from bench.tasks.structured import sha
from core.connectors.storage import LocalStorage
from core.db.spending import SpendingLedger
from core.eval.runtime import verify_runtime
from core.interventions.tf_runner import download_last_checkpoint, weights_digest
from scripts.fresh_answer_calibration import load as load_development
from scripts.fresh_answer_calibration import provider, save


def prepare(root, development, expected, cohort):
    dev, state = load_development(development, expected)
    decision_path = development / "power-decision.json"
    decision = json.loads(decision_path.read_text())
    if decision["protocol_sha256"] != expected or decision["status"] != "passed":
        raise ValueError("Development power gate did not pass")
    if any(not value.get("settled") for value in state.values()):
        raise ValueError("Settle all development jobs before fresh references")
    original = json.loads((cohort / "execution-plan.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    jobs = {}
    for name, seed in (("healthy", 4101), ("mild", 4102), ("strong", 4103)):
        label = "clean" if name == "healthy" else name
        source = cohort / (
            "evaluator-only/clean" if name == "healthy" else f"observed/{name}"
        )
        train, validation = source / "train.jsonl", source / "validation.jsonl"
        for role in ("train", "validation"):
            if sha(source / f"{role}.jsonl") != original["datasets"][label][role]:
                raise ValueError("Fresh data changed since preparation")
        rows = train.read_text().splitlines()
        random.Random(seed).shuffle(rows)
        shuffled = root / f"{name}-train.jsonl"
        shuffled.write_text("\n".join(rows) + "\n")
        jobs[name] = {
            "seed": seed,
            "train": str(shuffled.resolve()),
            "validation": str(validation.resolve()),
            "train_sha256": sha(shuffled),
            "validation_sha256": sha(validation),
            "reservation_usd": len(rows) * 306 * 0.4 / 1e6,
        }
    plan = {
        "phase": "fresh_references",
        "development_root": str(development.resolve()),
        "development_protocol_sha256": expected,
        "power_decision_sha256": sha(decision_path),
        "source_sha256": {
            str(Path(__file__).resolve()): sha(Path(__file__)),
            str(Path("scripts/fresh_answer_power.py").resolve()): sha(
                Path("scripts/fresh_answer_power.py")
            ),
        },
        "model": dev["model"],
        "recipe": original["recipe"],
        "evaluation_runtime": dev["evaluation_runtime"],
        "price": dev["price"],
        "jobs": jobs,
        "cohort": str(cohort.resolve()),
        "stopping": "Exactly one reference per condition. No retries after ambiguous submission. No reference selection by quality. No final-test evaluation here.",
    }
    save(root / "protocol.json", plan)
    digest = sha(root / "protocol.json")
    (root / "protocol.sha256").write_text(digest + "\n")
    print(digest)


def load(root, expected):
    if sha(root / "protocol.json") != expected:
        raise ValueError("Fresh reference protocol changed")
    plan = json.loads((root / "protocol.json").read_text())
    development = Path(plan["development_root"])
    dev, _ = load_development(development, plan["development_protocol_sha256"])
    if sha(development / "power-decision.json") != plan["power_decision_sha256"]:
        raise ValueError("Power decision changed")
    for path, digest in plan["source_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Reference execution source changed")
    for job in plan["jobs"].values():
        for role in ("train", "validation"):
            if sha(Path(job[role])) != job[f"{role}_sha256"]:
                raise ValueError("Fresh reference data changed")
    verify_runtime(plan["evaluation_runtime"], model=plan["model"])
    path = root / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    ledger = SpendingLedger(development / "spending.sqlite3", dev["cap_usd"])
    return plan, state, ledger


def submit(root, expected, env_file):
    plan, state, ledger = load(root, expected)
    tf = provider(env_file)
    for name, job in plan["jobs"].items():
        if name in state:
            if state[name].get("job_id"):
                print(name, "already submitted", state[name]["job_id"], flush=True)
                continue
            raise RuntimeError("Ambiguous fresh reference submission; reconcile first")
        ledger.reserve(
            f"fresh-reference-{name}", "fresh_reference", job["reservation_usd"]
        )
        state[name] = {"status": "submitting", "protocol_sha256": expected}
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
            suffix=f"ne-fresh-reference-{name}",
        )
        state[name].update(job_id=job_id, status="submitted")
        save(root / "state.json", state)
        print(name, job_id, flush=True)
    save(root / "spending.json", ledger.snapshot())


def collect(root, expected, env_file):
    plan, state, ledger = load(root, expected)
    tf, storage = provider(env_file), LocalStorage(root)
    for name, entry in state.items():
        if not entry.get("job_id"):
            raise RuntimeError("Unresolved submission")
        job = tf.job(entry["job_id"])
        entry["status"] = job.status
        save(root / "provider" / f"{name}.json", job.model_dump())
        if job.status == "succeeded":
            checkpoints = tf.checkpoints(job.id)
            save(
                root / "provider" / f"{name}-checkpoints.json",
                [c.model_dump() for c in checkpoints],
            )
            if not entry.get("adapter"):
                adapter = download_last_checkpoint(tf, storage, job.id, "checkpoints")
                entry.update(
                    adapter=str(adapter.resolve()),
                    weights_sha256=weights_digest(adapter),
                )
            if not entry.get("settled"):
                if type(job.trained_tokens) is not int or job.trained_tokens < 0:
                    raise ValueError("Provider token count unavailable")
                cost = job.trained_tokens * plan["price"]["usd_per_million"] / 1e6
                ledger.settle(f"fresh-reference-{name}", cost)
                entry.update(
                    settled=True,
                    trained_tokens=job.trained_tokens,
                    token_cost_estimate_usd=cost,
                )
        save(root / "state.json", state)
        print(name, job.status, flush=True)
    save(root / "spending.json", ledger.snapshot())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "collect"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.development_root, args.protocol_sha256, args.cohort)
    else:
        {"submit": submit, "collect": collect}[args.action](
            args.root, args.protocol_sha256, args.env_file
        )


if __name__ == "__main__":
    main()
