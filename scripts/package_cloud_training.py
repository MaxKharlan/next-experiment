"""Build an allowlisted, credential-free bundle for a prospective CUDA matrix."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from collections import Counter
from pathlib import Path

from bench.faults.semantic_data import drop_candidates_with_control, overlap_lines
from core.eval.runtime import file_digest
from scripts.prepare_data_fault_quality import save, write_rows

SOURCES = (
    "jobs/kaggle/train_lora.py",
    "bench/faults/labels.py",
    "bench/faults/recipe.py",
    "core/training/gradients.py",
    "core/training/parameters.py",
    "core/eval/data.py",
    "core/eval/generate.py",
    "core/eval/metrics.py",
    "core/eval/records.py",
    "core/eval/structured.py",
    "core/probes/level1/logits.py",
)


def package(kind, destination, qualification):
    decision = json.loads(qualification.read_text())
    if (
        decision.get("decision") != "pass"
        or not decision.get("token_hashes_equal")
        or not all(decision.get(name, {}).get("pass") for name in ("mps", "cpu"))
        or decision.get("precision")
        != {
            "matmul": "highest",
            "tf32": False,
            "NVIDIA_TF32_OVERRIDE": "0",
            "sdpa": "math only",
        }
    ):
        raise ValueError("Explicit FP32/math qualification must pass before packaging")
    destination.mkdir(parents=True, exist_ok=False)
    bundle = destination / "bundle"
    bundle.mkdir()
    code_root = Path("docs/results/pretrained-code-audit-20261020")
    code_plan = json.loads((code_root / "protocol.json").read_text())
    root = (
        code_root
        if kind == "code"
        else Path("docs/results/data-fault-quality-20261020")
    )
    if kind == "code":
        for name, digest in code_plan["source_sha256"].items():
            assert file_digest(name) == digest, name
        for role, digest in code_plan["datasets_sha256"].items():
            assert file_digest(root / f"{role}.jsonl") == digest, role
    else:
        preparation = json.loads((root / "preparation.json").read_text())
        for relative, digest in preparation["files_sha256"].items():
            assert file_digest(root / relative) == digest, relative
        repair = json.loads((root / "label-repair.json").read_text())
        for role in ("target", "control"):
            name = "labels-repair" if role == "target" else "labels-control"
            assert (
                file_digest(root / "training" / f"{name}.jsonl")
                == repair[f"{role}_sha256"]
            )
    for name in SOURCES:
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, path)
    shutil.copyfile("scripts/cloud_training_worker.py", bundle / "worker.py")
    shutil.copyfile("jobs/cloud/bootstrap.py", bundle / "main.py")
    bootstrap = bundle / "main.py"
    assert "PIP_CONSTRAINT" in bootstrap.read_text()
    runtime = code_plan["runtime"]
    runtime["packages"].update(
        {name: importlib.metadata.version(name) for name in ("pydantic", "numpy")}
    )
    (bundle / "dependencies.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in runtime["packages"].items())
    )
    data = bundle / "data"
    data.mkdir()
    for role in ("train", "validation", "check"):
        shutil.copyfile(root / f"{role}.jsonl", data / f"{role}.jsonl")
    seen = Counter()
    selected = []
    for line in (root / "validation.jsonl").read_text().splitlines():
        row = json.loads(line)
        if seen[row["source"]] < 2:
            selected.append(row)
            seen[row["source"]] += 1
    assert len(selected) == 16 and len(seen) == 8
    write_rows(data / "logit-validation.jsonl", selected)
    if kind == "code":
        arms = [
            {
                "name": fault or "healthy",
                "code_fault": fault,
                "train": "data/train.jsonl",
            }
            for fault in code_plan["arms"]
        ]
        plan = {
            "id": "pretrained-code-cuda-20261020",
            "predecessor_sha256": file_digest(code_root / "protocol.json"),
            "hp": code_plan["hp"],
            "seeds": code_plan["seeds"],
            "execution_amendment": "Complete the comparison as a separate full 15-cell CUDA cohort under the newly authorized AI Cloud budget. Existing two MPS adapter results remain supplemental, unchanged; remaining MPS cells are superseded by this execution plan. Same data, recipe and seeds; do not pool MPS and CUDA estimates or describe the two cohorts as independent samples. No old MPS training is rerun.",
        }
    else:
        training = data / "training"
        shutil.copytree(root / "training", training)
        arms = [
            {"name": path.stem, "train": str(path.relative_to(bundle))}
            for path in sorted(training.glob("*.jsonl"))
        ]
        train = [
            json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()
        ]
        extra = ["data/validation.jsonl"]
        for percent in (25, 50):
            name = f"overlap-validation-{percent}"
            contaminated = [
                json.loads(line)
                for line in (root / f"{name}.jsonl").read_text().splitlines()
            ]
            indices = overlap_lines(train, contaminated)
            target, control = drop_candidates_with_control(contaminated, indices, 6112)
            for suffix, rows in (
                ("", contaminated),
                ("-dedup", target),
                ("-control", control),
            ):
                relative = f"{name}{suffix}.jsonl"
                write_rows(data / relative, rows)
                extra.append("data/" + relative)
        for arm in arms:
            if arm["name"] == "healthy":
                arm["extra_evaluations"] = extra
        plan = {
            "id": "data-fault-quality-cuda-20261020",
            "preparation_sha256": file_digest(root / "preparation.json"),
            "observed_repair_sha256": file_digest(root / "label-repair.json"),
            "hp": preparation["hp"],
            "seeds": preparation["seeds"],
            "scope": preparation["stopping"],
        }
    plan.update(
        id=destination.name,
        runtime=runtime,
        arms=arms,
        evaluate_base=True,
        device="cuda",
        timeout_hours=2,
        reservation_usd=6,
        purpose="Development quality and paired descriptive effects; no final-test use or fitted deployment likelihood.",
        logit_diagnostics="First two validation queries per class, 16 total; same references for every arm. Descriptive NLL/EOS measurements, no fitted thresholds.",
        launch_gate="Explicit FP32/math SDPA qualification passed against both CPU and MPS at the unchanged 1e-4 limits.",
        qualification_path=str(qualification),
        qualification_sha256=file_digest(qualification),
        precision_amendment="Initial default-CUDA bundles were never submitted because their numerical gate failed. This new bundle explicitly disables TF32 and fused SDPA before training. Data, seeds, recipes and endpoints are unchanged. No training result has been observed for either CUDA matrix.",
        source_packager_sha256=file_digest(__file__),
        bundle_files_sha256={
            str(path.relative_to(bundle)): file_digest(path)
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        },
        failure_policy="No automatic retry of started training. Every completed cell is persisted; infrastructure failures remain explicit. Never change a recipe to improve an observed effect.",
    )
    save(bundle / "experiment.json", plan)
    save(destination / "protocol.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("code", "data"))
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--qualification", required=True, type=Path)
    args = parser.parse_args()
    p = package(args.kind, args.destination, args.qualification)
    print(json.dumps({"id": p["id"], "cells": len(p["arms"]) * len(p["seeds"])}))
