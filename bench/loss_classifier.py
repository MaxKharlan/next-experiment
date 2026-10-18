"""B9: an offline nearest-centroid baseline for simulated loss curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

FEATURES = (
    "train_slope",
    "valid_slope",
    "final_gap",
    "valid_min_position",
    "train_tail_change",
)
LABELS = {"healthy", "lr_too_low", "lora_rank_too_small", "too_many_epochs"}


def curve_features(records):
    """Normalize each curve's step coordinate; consume losses only."""
    curves = {}
    for key in ("train_loss", "valid_loss"):
        points = [(r["step"], r[key]) for r in records if key in r]
        if len(points) < 2:
            raise ValueError(f"At least two {key} observations required")
        values = np.asarray(points, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Steps and losses must be finite and nonnegative")
        x, y = values.T
        if (np.diff(x) <= 0).any():
            raise ValueError("Steps must increase strictly within each loss curve")
        x = (x - x[0]) / (x[-1] - x[0])
        slope = float(
            np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean())
        )
        curves[key] = (x, y, slope)
    tx, train, ts = curves["train_loss"]
    vx, valid, vs = curves["valid_loss"]
    tail = train[tx >= 0.8]
    if len(tail) < 2:
        tail = train[-2:]
    return [
        ts,
        vs,
        float(valid[-1] - train[-1]),
        float(vx[np.argmin(valid)]),
        float(tail[-1] - tail[0]),
    ]


def fit(rows):
    x = np.asarray([r["features"] for r in rows], dtype=float)
    if (
        not rows
        or x.ndim != 2
        or x.shape[1] != len(FEATURES)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid training feature matrix")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    z = (x - mean) / scale
    labels = sorted({r["label"] for r in rows})
    if len(labels) < 2:
        raise ValueError("At least two recipe classes required")
    return {
        "version": 1,
        "features": list(FEATURES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "centroids": {
            label: z[[r["label"] == label for r in rows]].mean(axis=0).tolist()
            for label in labels
        },
    }


def predict(model, features):
    x = np.asarray(features, dtype=float)
    if x.shape != (len(FEATURES),) or not np.isfinite(x).all():
        raise ValueError("Invalid prediction features")
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return min(
        model["centroids"],
        key=lambda label: (
            float(np.square(z - model["centroids"][label]).sum()),
            label,
        ),
    )


def evaluate(rows):
    seeds = sorted({r["seed"] for r in rows})
    labels = sorted({r["label"] for r in rows})
    if len(seeds) < 2:
        raise ValueError("At least two seeds required for grouped evaluation")
    predictions, folds = [], []
    for seed in seeds:
        train = [r for r in rows if r["seed"] != seed]
        test = [r for r in rows if r["seed"] == seed]
        if {r["label"] for r in train} != set(labels):
            raise ValueError("Every fold must retain all recipe classes in training")
        model = fit(train)
        folds.append(
            {
                "excluded_seed": seed,
                "train_ids": [r["id"] for r in train],
                "test_ids": [r["id"] for r in test],
                "model": model,
            }
        )
        for row in test:
            predictions.append(
                {
                    "id": row["id"],
                    "seed": seed,
                    "truth": row["label"],
                    "predicted": predict(model, row["features"]),
                }
            )
    matrix = {label: {other: 0 for other in labels} for label in labels}
    for row in predictions:
        matrix[row["truth"]][row["predicted"]] += 1
    return {
        "protocol": "leave-one-seed-out; scaler fitted inside each fold",
        "n": len(rows),
        "accuracy": sum(r["truth"] == r["predicted"] for r in predictions) / len(rows),
        "balanced_accuracy": float(
            np.mean([matrix[k][k] / sum(matrix[k].values()) for k in labels])
        ),
        "confusion": matrix,
        "predictions": predictions,
        "folds": folds,
    }


def load_runs(root):
    rows, seen = [], set()
    for path in sorted(root.glob("*/summary.json")):
        summary_raw = path.read_bytes()
        summary = json.loads(summary_raw)
        label = summary["fault"] or "healthy"
        if label not in LABELS or type(summary["seed"]) is not int:
            raise ValueError("Unsupported label or seed")
        identity = (summary["model"], label, summary["strength"], summary["seed"])
        if identity in seen:
            raise ValueError("Duplicate recipe/seed run")
        seen.add(identity)
        raw = path.with_name("metrics.jsonl").read_bytes()
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        rows.append(
            {
                "id": path.parent.name,
                "label": label,
                "seed": summary["seed"],
                "strength": summary["strength"],
                "model": summary["model"],
                "features": curve_features(records),
                "summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
                "metrics_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not rows:
        raise ValueError("No run summaries found")
    if len({r["model"] for r in rows}) != 1:
        raise ValueError("Evaluate one base model at a time")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory already exists")
    rows = load_runs(args.runs)
    report = evaluate(rows)
    model = fit(rows)
    args.out.mkdir(parents=True, exist_ok=False)
    for path in sorted(args.runs.glob("*/summary.json")):
        target = args.out / "inputs" / path.parent.name
        target.mkdir(parents=True)
        shutil.copyfile(path, target / "summary.json")
        shutil.copyfile(path.with_name("metrics.jsonl"), target / "metrics.jsonl")
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["scope"] = "Simulated recipe classification; not verified failure diagnosis"
    for name, data in (("rows", rows), ("evaluation", report), ("model", model)):
        (args.out / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"{len(rows)} runs; grouped accuracy={report['accuracy']:.3f}; balanced={report['balanced_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
