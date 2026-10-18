"""Fit measured training-stage durations with a preselected validation condition."""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np


def fit_model(rows):
    if len({r["model"] for r in rows}) != 1:
        raise ValueError("Fit one base model at a time")
    train, validation = [], []
    for row in rows:
        if row["status"] != "succeeded":
            raise ValueError("Cannot fit failed jobs")
        seconds = row["seconds"]
        duration = seconds["training_completed"] - seconds["training_started"]
        tokens = row["trained_tokens"]
        epochs = row["epochs"]
        if not all(math.isfinite(v) and v > 0 for v in (duration, tokens, epochs)):
            raise ValueError("Invalid duration, token count or epochs")
        sample = (row, [1, tokens / 1000, epochs - 1], duration)
        (validation if (row["examples"], epochs) == (500, 2) else train).append(sample)
    if len(train) < 4 or not validation:
        raise ValueError(
            "Need four fitting conditions and the 500-example/2-epoch holdback"
        )
    x = np.array([sample[1] for sample in train])
    if np.linalg.matrix_rank(x) != 3:
        raise ValueError("Unidentifiable duration model")
    y = np.array([sample[2] for sample in train])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    checks = []
    for row, features, actual in validation:
        prediction = float(np.dot(features, coefficients))
        checks.append(
            {
                "job_id": row["job_id"],
                "actual_seconds": actual,
                "predicted_seconds": prediction,
                "absolute_relative_error": abs(prediction - actual) / actual,
            }
        )
    return {
        "model": rows[0]["model"],
        "scope": "observed training stage; provider capacity waiting may be included",
        "features": ["intercept", "trained_tokens/1000", "epochs-1"],
        "coefficients": coefficients.tolist(),
        "fixed_stages_seconds": {
            "datasets_s": statistics.median(
                r["seconds"]["datasets"] for r, _, _ in train
            ),
            "startup_s": statistics.median(
                r["seconds"]["training_started"] - r["seconds"]["datasets"]
                for r, _, _ in train
            ),
            "finalize_s": statistics.median(
                r["seconds"]["finished"] - r["seconds"]["training_completed"]
                for r, _, _ in train
            ),
        },
        "fit_job_ids": [r["job_id"] for r, _, _ in train],
        "fit_rmse_seconds": float(np.sqrt(np.mean((x @ coefficients - y) ** 2))),
        "validation": checks,
        "nonnegative_coefficients": bool((coefficients >= 0).all()),
        "cost_basis": "trained_tokens multiplied by recorded rate; not a billing receipt",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.directory / "plan.json").read_text())
    rows = [
        json.loads((args.directory / f"{r['run']}.json").read_text())
        for r in plan["runs"]
    ]
    reports = [
        fit_model([r for r in rows if r["model"] == model])
        for model in sorted({r["model"] for r in rows})
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(
            {"models": reports, "recorded_cost_usd": sum(r["cost_usd"] for r in rows)},
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
