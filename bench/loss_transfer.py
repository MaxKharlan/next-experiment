"""Evaluate a frozen local loss classifier on saved Token Factory cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bench.cases import load_cases
from bench.loss_classifier import FEATURES, LABELS, curve_features, predict


def validate_model(model):
    if model.get("version") != 1 or model.get("features") != list(FEATURES):
        raise ValueError("Unsupported model or feature order")
    centroids = model.get("centroids", {})
    if len(centroids) < 2 or not set(centroids) <= LABELS:
        raise ValueError("Invalid recipe classes")
    for value in [model["mean"], model["scale"], *centroids.values()]:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (len(FEATURES),) or not np.isfinite(vector).all():
            raise ValueError("Invalid model vector")
    if (np.asarray(model["scale"]) <= 0).any():
        raise ValueError("Model scales must be positive")


def evaluate(cases, model):
    validate_model(model)
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("Expected nonempty unique cases")
    jobs = [c.job_id for c in cases]
    if any(not j.startswith("ftjob-") for j in jobs) or len(set(jobs)) != len(jobs):
        raise ValueError("Expected unique Token Factory jobs")
    rows = []
    for case in cases:
        if case.split not in {"dev", "heldout"} or len(case.truth) > 1:
            raise ValueError("Unsupported split or compound truth")
        truth = case.truth[0] if case.truth else "healthy"
        if truth not in model["centroids"]:
            raise ValueError(f"Model has no class for {truth}")
        points = case.case.get("checkpoints", [])
        # Epoch is the sampling coordinate; the extractor normalizes it.
        records = [
            {
                "step": p["epoch"],
                "train_loss": p["train_loss"],
                "valid_loss": p["valid_loss"],
            }
            for p in points
        ]
        features = curve_features(records)
        rows.append(
            {
                "case": case.id,
                "job_id": case.job_id,
                "split": case.split,
                "truth": truth,
                "predicted": predict(model, features),
                "features": features,
                "checkpoint_count": len(points),
            }
        )
    groups = {}
    for split in sorted({r["split"] for r in rows}):
        selected = [r for r in rows if r["split"] == split]
        labels = sorted(model["centroids"])
        matrix = {k: {v: 0 for v in labels} for k in labels}
        for r in selected:
            matrix[r["truth"]][r["predicted"]] += 1
        supported = [k for k in labels if sum(matrix[k].values())]
        groups[split] = {
            "n": len(selected),
            "accuracy": sum(r["truth"] == r["predicted"] for r in selected)
            / len(selected),
            "balanced_accuracy_observed_classes": float(
                np.mean([matrix[k][k] / sum(matrix[k].values()) for k in supported])
            ),
            "supported_classes": supported,
            "confusion": matrix,
        }
    return {
        "version": 1,
        "refitted": False,
        "scope": "Exploratory transfer on previously inspected cases",
        "rows": rows,
        "splits": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists")
    raw = args.model.read_bytes()
    cases = load_cases("dev") + load_cases("heldout")
    report = evaluate(cases, json.loads(raw))
    report["model_sha256"] = hashlib.sha256(raw).hexdigest()
    report["case_sha256"] = {
        c.id: hashlib.sha256(json.dumps(asdict(c), sort_keys=True).encode()).hexdigest()
        for c in cases
    }
    report["source_sha256"] = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in ["bench/loss_classifier.py", "bench/loss_transfer.py"]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
