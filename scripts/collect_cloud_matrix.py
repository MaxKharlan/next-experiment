"""Collect terminal matrix artifacts and verify their frozen provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from statistics import mean, stdev

from core.config import Settings
from core.connectors.compute import JobSpec, ServerlessJobs
from core.connectors.storage import S3Storage
from core.db.spending import SpendingLedger
from core.eval.data import read_examples
from core.eval.metrics import exact_match
from core.eval.runtime import file_digest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def safe_relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe artifact path")
    return path


def verify_predictions(rows, examples):
    if len(rows) != len(examples):
        raise ValueError("Prediction count differs from frozen check set")
    for row, example in zip(rows, examples, strict=True):
        if (row["line"], row["expected"], row["source"]) != (
            example.line,
            example.expected,
            example.source,
        ):
            raise ValueError("Prediction identity differs from frozen check set")
        score = 0.0 if row["hit_limit"] else exact_match(row["answer"], row["expected"])
        if row["score"] != score:
            raise ValueError("Stored exact-match score differs from recomputation")
    return {
        "examples": len(rows),
        "exact_match": mean(row["score"] for row in rows),
        "eos_share": mean(row["stopped_on_eos"] for row in rows),
        "limit_share": mean(row["hit_limit"] for row in rows),
    }


def descriptive(values):
    return {
        "n": len(values),
        "values": values,
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else None,
    }


def analyze(plan, result, artifacts, bundle):
    if result["id"] != plan["id"]:
        raise ValueError("Result belongs to another experiment")
    runtime = result["runtime"]
    if runtime["protocol_sha256"] != file_digest(bundle / "experiment.json"):
        raise ValueError("Remote protocol digest differs")
    if runtime["model_revision"] != plan["runtime"]["revision"]:
        raise ValueError("Remote model revision differs")
    for package, version in plan["runtime"]["packages"].items():
        if runtime["packages"][package].split("+")[0] != version:
            raise ValueError("Remote package version differs")
    if runtime["precision"] != {
        "matmul": "highest",
        "tf32": False,
        "NVIDIA_TF32_OVERRIDE": "0",
        "sdpa": "math only",
    }:
        raise ValueError("Remote numerical configuration differs")
    expected = {
        f"{arm['name']}-s{seed}" for arm in plan["arms"] for seed in plan["seeds"]
    }
    if set(result["cells"]) - expected:
        raise ValueError("Unexpected matrix cells")
    if result["status"] == "complete" and set(result["cells"]) != expected:
        raise ValueError("Completed matrix is missing cells")
    examples = read_examples(bundle / "data/check.jsonl")
    base = None
    if "base" in result:
        base = verify_predictions(
            json.loads((artifacts / "base/predictions.json").read_text()), examples
        )
        if any(abs(value - result["base"][key]) > 1e-12 for key, value in base.items()):
            raise ValueError("Base metrics differ from saved predictions")
    verified = {}
    for name, metrics in result["cells"].items():
        directory = artifacts / name
        manifest = json.loads((directory / "manifest.json").read_text())
        for relative, digest in manifest.items():
            path = directory.joinpath(*safe_relative(relative).parts)
            if file_digest(path) != digest:
                raise ValueError(f"Artifact digest differs: {name}/{relative}")
        if (
            file_digest(directory / "adapter/adapter_model.safetensors")
            != metrics["weights_sha256"]
        ):
            raise ValueError("Adapter differs from reported weights")
        if file_digest(directory / "summary.json") != metrics["summary_sha256"]:
            raise ValueError("Training summary digest differs")
        rows = json.loads((directory / "predictions.json").read_text())
        computed = verify_predictions(rows, examples)
        for key, value in computed.items():
            if abs(value - metrics[key]) > 1e-12:
                raise ValueError("Reported metric differs from saved predictions")
        summary = json.loads((directory / "summary.json").read_text())
        arm_name, seed_text = name.rsplit("-s", 1)
        arm = next(a for a in plan["arms"] if a["name"] == arm_name)
        if summary["seed"] != int(seed_text):
            raise ValueError("Training seed differs")
        if summary["code_fault"] != arm.get("code_fault"):
            raise ValueError("Training fault differs")
        if summary["hyperparameters"] != plan["hp"]:
            raise ValueError("Training recipe differs")
        verified[name] = {
            **computed,
            "weights_sha256": metrics["weights_sha256"],
            "training": {
                key: summary[key]
                for key in (
                    "wall_seconds",
                    "steps",
                    "train_examples",
                    "input_tokens",
                    "supervised_tokens",
                    "answer_tokens",
                )
            },
        }
        for relative in arm.get("extra_evaluations", []):
            extra = directory / "extra" / Path(relative).stem
            rows = json.loads((extra / "predictions.json").read_text())
            computed = verify_predictions(rows, read_examples(bundle / relative))
            stored = json.loads((extra / "evaluation.json").read_text())
            if any(abs(value - stored[key]) > 1e-12 for key, value in computed.items()):
                raise ValueError("Extra evaluation metrics differ")
            verified[name].setdefault("extra_evaluations", {})[relative] = computed
    arms = {}
    for arm in plan["arms"]:
        cells = [
            verified[f"{arm['name']}-s{seed}"]
            for seed in plan["seeds"]
            if f"{arm['name']}-s{seed}" in verified
        ]
        if cells:
            arms[arm["name"]] = {
                metric: descriptive([cell[metric] for cell in cells])
                for metric in ("exact_match", "eos_share", "limit_share")
            }
    pairs = {}
    for arm in plan["arms"]:
        name = arm["name"]
        if name == "healthy":
            continue
        seeds = [
            seed
            for seed in plan["seeds"]
            if f"healthy-s{seed}" in verified and f"{name}-s{seed}" in verified
        ]
        if seeds:
            pairs[f"healthy-minus-{name}"] = {
                "seeds": seeds,
                **descriptive(
                    [
                        verified[f"healthy-s{seed}"]["exact_match"]
                        - verified[f"{name}-s{seed}"]["exact_match"]
                        for seed in seeds
                    ]
                ),
            }
    for target, control in (
        ("labels-repair", "labels-50"),
        ("labels-control", "labels-50"),
        ("labels-repair", "labels-control"),
        ("shift-repair", "shift-strong"),
        ("shift-control", "shift-strong"),
        ("shift-repair", "shift-control"),
        ("healthy", "foreign-control"),
    ):
        seeds = [
            seed
            for seed in plan["seeds"]
            if f"{target}-s{seed}" in verified and f"{control}-s{seed}" in verified
        ]
        if seeds:
            pairs[f"{target}-minus-{control}"] = {
                "seeds": seeds,
                **descriptive(
                    [
                        verified[f"{target}-s{seed}"]["exact_match"]
                        - verified[f"{control}-s{seed}"]["exact_match"]
                        for seed in seeds
                    ]
                ),
            }
    return {
        "status": result["status"],
        "base": base,
        "verified_cells": verified,
        "missing_cells": sorted(expected - verified.keys()),
        "arms": arms,
        "paired_exact_match_differences": pairs,
        "scope": "Descriptive development comparisons. Seed SD measures training variation on one fixed check set; no independent population or confirmatory claim.",
        "cost": "Reported training tokens and wall time are computational measurements. Cloud reservations remain unsettled until billing evidence.",
    }


def collect(root, destination, env_file):
    execution = json.loads((root / "execution-plan.json").read_text())
    spec_data = execution["spec"]
    spec_data["entry"] = Path(spec_data["entry"])
    spec = JobSpec(**spec_data)
    client = ServerlessJobs(
        SpendingLedger(
            "artifacts/remaining-queue-20261020/ai-cloud-spending.sqlite3", 30
        ),
        root / "receipts",
        cli="/Users/harlanmax/.nebius/bin/nebius",
    )
    status = client.status(execution["job_id"])
    prefix = f"runs/{spec.name}/"
    if (
        status["metadata"]["name"] != spec.name
        or status["metadata"]["parent_id"] != spec.project
        or status["status"]["state"] not in {"COMPLETED", "FAILED", "CANCELLED"}
        or not any(
            v.get("source") == spec.output_bucket_id
            and v.get("source_path") == prefix
            and v.get("container_path") == "/outputs"
            for v in status["spec"].get("volumes", [])
        )
    ):
        raise ValueError("Require terminal job with matching identity and output mount")
    storage = S3Storage.from_settings(Settings(_env_file=env_file))
    if storage.bucket != spec.output_bucket:
        raise ValueError("Storage bucket differs from job")
    keys = storage.list_keys(prefix)
    downloaded = {}
    for key in keys:
        if not key.startswith(prefix):
            raise ValueError("Storage returned a key outside the job")
        relative = safe_relative(key.removeprefix(prefix))
        if relative.as_posix() == "input.zip":
            continue
        raw = storage.get_bytes(key)
        path = destination.joinpath(*relative.parts)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("Previously collected terminal artifact changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        downloaded[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
    save(root / "status.json", status)
    save(
        root / "collection.json",
        {"files_sha256": downloaded, "job_id": execution["job_id"]},
    )
    if (destination / "result.json").exists():
        result = json.loads((destination / "result.json").read_text())
        plan = json.loads((root / "protocol.json").read_text())
        if "studies" in plan:
            if result["protocol_sha256"] != file_digest(root / "protocol.json"):
                raise ValueError("Combined protocol digest differs")
            report = {"status": result["status"], "studies": {}}
            for study in plan["studies"]:
                name = study["name"]
                if (destination / name / "result.json").exists():
                    sub_result = json.loads(
                        (destination / name / "result.json").read_text()
                    )
                    sub_bundle = root / "bundle" / name
                    sub_plan = json.loads((sub_bundle / "experiment.json").read_text())
                    report["studies"][name] = analyze(
                        sub_plan, sub_result, destination / name, sub_bundle
                    )
                else:
                    report["studies"][name] = {"status": "no_result"}
        else:
            report = analyze(plan, result, destination, root / "bundle")
        if status["status"]["state"] == "COMPLETED" and result["status"] != "complete":
            raise ValueError("Completed job did not finish the matrix")
        save(root / "result.json", result)
        save(root / "analysis.json", report)
        for path in destination.rglob("*.json"):
            relative = path.relative_to(destination)
            save(root / "measurements" / relative, json.loads(path.read_text()))
    return {
        "job": execution["job_id"],
        "state": status["status"]["state"],
        "files": len(downloaded),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.destination, args.env_file)))
