"""Freeze code and data studies in one final bounded cloud allocation."""

import json
import shutil
from pathlib import Path

from core.eval.data import read_examples
from core.eval.runtime import file_digest
from scripts.collect_cloud_matrix import save, verify_predictions
from scripts.package_cloud_training import package

ROOT = Path("docs/results/cloud-matrices-export-20261020")

ORCHESTRATOR = '''"""Execute each frozen study once, with separate output and time limits."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).parent
plan = json.loads((root / "experiment.json").read_text())
for name, digest in plan["bundle_files_sha256"].items():
    with (root / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
output = Path(os.environ["NEBIUS_OUTPUT_DIR"])
results = {"status": "running", "studies": {}, "protocol_sha256": hashlib.sha256((root / "experiment.json").read_bytes()).hexdigest()}
started = time.monotonic()
for study in plan["studies"]:
    destination = output / study["name"]
    destination.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "NEBIUS_OUTPUT_DIR": str(destination)}
    try:
        run = subprocess.run([sys.executable, str(root / study["worker"])], env=env, timeout=study["timeout_seconds"])
        entry = {"returncode": run.returncode, "timed_out": False}
    except subprocess.TimeoutExpired:
        entry = {"returncode": None, "timed_out": True}
    entry["elapsed_total_seconds"] = time.monotonic() - started
    results["studies"][study["name"]] = entry
    (output / "result.json").write_text(json.dumps(results, indent=2) + "\\n")
results["status"] = "complete" if all(r["returncode"] == 0 for r in results["studies"].values()) else "partial_or_failed"
(output / "result.json").write_text(json.dumps(results, indent=2) + "\\n")
sys.exit(0 if results["status"] == "complete" else 1)
'''


def main():
    ROOT.mkdir(exist_ok=False)
    bundle = ROOT / "bundle"
    bundle.mkdir()
    studies = []
    for kind, prefix, timeout in (
        ("code", "pretrained-code", 1800),
        ("data", "data-fault", 3600),
    ):
        previous = Path(f"docs/results/{prefix}-cuda-cleanpip-20261020")
        target = Path(f"docs/results/{prefix}-cuda-export-20261020")
        plan = package(
            kind, target, Path("docs/results/cloud-precision-20261020/decision.json")
        )
        old = json.loads((previous / "protocol.json").read_text())
        for name, digest in old["bundle_files_sha256"].items():
            if name != "worker.py":
                assert file_digest(target / "bundle" / name) == digest, name
        plan["export_amendment"] = {
            "previous_protocol_sha256": file_digest(previous / "protocol.json"),
            "reason": "Object-store mount rejected copytree POSIX metadata after base generation; no training started. Copy bytes only. Fix BatchEncoding length guard; independent recount already found every input <=169 tokens. Each study gets a distinct temporary/output directory.",
            "failed_job_id": "aijob-e00s96et8m5pkns9m0",
            "study_timeout_seconds": timeout,
            "shared_job_timeout_hours": 2,
            "budget": "Final shared $6 reservation inside the $30 cloud cap, including retained previous holds. No extra paid retries.",
        }
        if kind == "code":
            saved = Path("artifacts/remaining-queue-20261020/code-cleanpip-collected")
            rows = json.loads((saved / "base/predictions.json").read_text())
            metrics = json.loads((saved / "base/evaluation.json").read_text())
            computed = verify_predictions(
                rows, read_examples(target / "bundle/data/check.jsonl")
            )
            assert all(
                abs(value - metrics[key]) < 1e-12 for key, value in computed.items()
            )
            runtime = json.loads((saved / "runtime.json").read_text())
            assert runtime["protocol_sha256"] == file_digest(previous / "protocol.json")
            assert runtime["model_revision"] == plan["runtime"]["revision"]
            for name, value in plan["runtime"]["packages"].items():
                assert runtime["packages"][name].split("+")[0] == value
            shutil.copytree(saved / "base", target / "bundle/data/precomputed-base")
            plan.update(
                evaluate_base=False,
                reuse_base="data/precomputed-base",
                base_reused_from={
                    "job_id": "aijob-e00s96et8m5pkns9m0",
                    "runtime_sha256": file_digest(saved / "runtime.json"),
                    "predictions_sha256": file_digest(saved / "base/predictions.json"),
                    "reason": "The completed base generation is reused unchanged; no base generation retry.",
                },
            )
        plan["bundle_files_sha256"] = {
            p.relative_to(target / "bundle").as_posix(): file_digest(p)
            for p in sorted((target / "bundle").rglob("*"))
            if p.is_file() and p != target / "bundle/experiment.json"
        }
        save(target / "bundle/experiment.json", plan)
        save(target / "protocol.json", plan)
        shutil.copytree(target / "bundle", bundle / kind)
        studies.append(
            {"name": kind, "worker": f"{kind}/worker.py", "timeout_seconds": timeout}
        )
    assert (bundle / "code/dependencies.txt").read_bytes() == (
        bundle / "data/dependencies.txt"
    ).read_bytes()
    shutil.copyfile("jobs/cloud/bootstrap.py", bundle / "main.py")
    shutil.copyfile(bundle / "code/dependencies.txt", bundle / "dependencies.txt")
    (bundle / "worker.py").write_text(ORCHESTRATOR)
    plan = {
        "id": ROOT.name,
        "studies": studies,
        "purpose": "One final infrastructure-corrected execution of two frozen development matrices. Scientific data, model, recipe, seeds and endpoints remain unchanged. Training has not previously started.",
        "timeout_hours": 2,
        "reservation_usd": 6,
        "stopping": "Each study executes once; a failed or timed-out study is retained and never retried. Independent remaining study may continue within its own time limit. No additional cloud jobs within this cap.",
        "bundle_files_sha256": {
            p.relative_to(bundle).as_posix(): file_digest(p)
            for p in sorted(bundle.rglob("*"))
            if p.is_file()
        },
    }
    save(bundle / "experiment.json", plan)
    save(ROOT / "protocol.json", plan)
    print(json.dumps({"id": plan["id"], "cells": 51, "reserve_usd": 6}))


if __name__ == "__main__":
    main()
