"""Correct the omitted LoRA flag; retain invalid v1 attempts and their holds."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from core.eval.runtime import file_digest
from scripts import execute_ticket_ledger as original
from scripts.fresh_answer_calibration import provider, save


def check_accepted(job, model, training_file, kwargs):
    observed = job.model_dump()
    hp = observed["hyperparameters"]
    if hp.get("lora") is not True:
        raise ValueError("Provider did not accept LoRA mode")
    for key, value in kwargs["hyperparameters"].items():
        if hp.get(key) != value:
            raise ValueError(f"Provider changed hyperparameter {key}")
    for key, value in {
        "model": model,
        "training_file": training_file,
        "validation_file": kwargs["validation_file"],
        "seed": kwargs["seed"],
    }.items():
        if observed.get(key) != value:
            raise ValueError(f"Provider changed {key}")
    return observed


def prepare(root, snapshot, predecessor):
    previous, state, prior_digest = original.load(predecessor)
    cancellations = json.loads(
        (predecessor / "wrong-mode-cancellation.json").read_text()
    )
    if set(cancellations) != set(previous["jobs"]) or any(
        row["after"]["status"] != "cancelled" for row in cancellations.values()
    ):
        raise ValueError("Every invalid attempt must be confirmed cancelled first")
    original.prepare(root, snapshot)
    plan = json.loads((root / "protocol.json").read_text())
    plan["id"] = "ticket-ledger-execution-20261020-v2"
    plan["recipe"]["lora"] = True
    plan["source_sha256"]["scripts/execute_ticket_ledger_v2.py"] = file_digest(__file__)
    plan["correction"] = {
        "predecessor_protocol_sha256": prior_digest,
        "cancellation_evidence_sha256": file_digest(
            predecessor / "wrong-mode-cancellation.json"
        ),
        "reason": "v1 omitted lora=true and was accepted as Full SFT; all attempts cancelled with holds retained. Correct planned LoRA mode; no data/gate/seed changes.",
        "predecessor_root": str(predecessor.resolve()),
    }
    plan["provider_files"] = {}
    for name, entry in state.items():
        job = previous["jobs"][name]
        if job["train_sha256"] != plan["jobs"][name]["train_sha256"]:
            raise ValueError("Correction changed training data")
        plan["provider_files"][job["train_sha256"]] = entry["training_file"]
    plan["provider_files"][file_digest(plan["validation"])] = next(
        iter(state.values())
    )["validation_file"]
    plan["stopping"] = (
        "Check every accepted job's LoRA flag, full recipe, model, files and seed before submitting the next. A mismatch stops and cancels that invalid attempt. No further execution retries; original scientific gates unchanged."
    )
    save(root / "protocol.json", plan)
    (root / "protocol.sha256").write_text(file_digest(root / "protocol.json") + "\n")


class CheckedProvider:
    def __init__(self, tf, root, plan):
        self.tf, self.root, self.plan = tf, root, plan

    def upload(self, path):
        return self.plan["provider_files"][file_digest(path)]

    def create_job(self, model, training_file, **kwargs):
        kwargs["suffix"] = kwargs["suffix"].replace("ne-ledger-", "ne-ledger-v2-", 1)
        job_id = self.tf.create_job(model, training_file, **kwargs)
        job = self.tf.job(job_id)
        save(self.root / "accepted" / f"{job_id}.json", job.model_dump())
        try:
            check_accepted(job, model, training_file, kwargs)
        except ValueError as error:
            save(
                self.root / "acceptance-rejection.json",
                {"job_id": job_id, "error": str(error)},
            )
            if job.status not in {"succeeded", "failed", "cancelled"}:
                self.tf.client.fine_tuning.jobs.cancel(job_id)
            raise
        return job_id


def submit(root, env):
    plan, _, _ = original.load(root)
    tf = CheckedProvider(provider(env), root, plan)
    old_provider = original.provider
    try:
        original.provider = lambda _: tf
        original.submit(root, env)
    finally:
        original.provider = old_provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "submit", "collect", "score", "analyze")
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    with (args.root.parent / "ticket-ledger-v2.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "prepare":
            prepare(args.root, args.snapshot, args.predecessor)
        elif args.action == "submit":
            submit(args.root, args.env_file)
        elif args.action == "collect":
            original.collect(args.root, args.env_file)
        elif args.action == "score":
            original.score(args.root)
        else:
            original.analyze(args.root)


if __name__ == "__main__":
    main()
