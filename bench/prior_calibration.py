"""Fit and cross-validate temperature on saved Super priors, before any probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from bench.calibration import reliability
from bench.cases import load_cases
from core.agent.prior_calibration import distribution, scale_distribution
from core.agent.schemas import HypothesisList


def metrics(rows, temperature):
    samples, losses, briers = [], [], []
    for row in rows:
        p = scale_distribution(row["probabilities"], temperature)
        truth = row["target"]
        winner = min(p, key=lambda k: (-p[k], k))
        samples.append((p[winner], winner == truth))
        losses.append(-math.log(max(1e-15, p[truth])))
        briers.append(sum((v - (k == truth)) ** 2 for k, v in p.items()))
    return {
        "nll": sum(losses) / len(rows),
        "multiclass_brier": sum(briers) / len(rows),
        "reliability": reliability(samples),
    }


def fit(rows):
    if not rows:
        raise ValueError("Empty calibration set")
    # NLL is convex in inverse temperature. Golden-section search on [0.05, 20].
    lo, hi = 0.05, 20.0
    for _ in range(90):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if metrics(rows, 1 / a)["nll"] < metrics(rows, 1 / b)["nll"]:
            hi = b
        else:
            lo = a
    choices = [0.05, 1.0, 20.0, 1 / ((lo + hi) / 2)]
    return min(choices, key=lambda t: (metrics(rows, t)["nll"], abs(math.log(t))))


def audit(rows):
    ids = [r["case"] for r in rows]
    if len(rows) < 3 or len(ids) != len(set(ids)):
        raise ValueError("Need at least three unique dev cases")
    temperature = fit(rows)
    folds = []
    for row in rows:
        training = [r for r in rows if r["case"] != row["case"]]
        t = fit(training)
        folds.append(
            {
                "case": row["case"],
                "training_cases": [r["case"] for r in training],
                "temperature": t,
                "before": metrics([row], 1),
                "after": metrics([row], t),
            }
        )
    return {
        "version": 1,
        "scope": "proposed classes plus unlisted mass",
        "temperature": temperature,
        "at_search_boundary": temperature < 0.050001 or temperature > 19.999999,
        "leave_one_case_out_mean": {
            stage: {
                metric: sum(f[stage][metric] for f in folds) / len(folds)
                for metric in ("nll", "multiclass_brier")
            }
            for stage in ("before", "after")
        },
        "fit_cases": ids,
        "before": metrics(rows, 1),
        "fitted_dev": metrics(rows, temperature),
        "leave_one_case_out": folds,
        "generalization_claim": False,
        "rows": rows,
    }


def load_rows(draws, split):
    rows = []
    for case in load_cases(split):
        path = draws / f"{case.id}--draw-1.json"
        raw = path.read_bytes()
        hypotheses = HypothesisList.model_validate_json(raw).hypotheses
        probabilities = distribution(hypotheses)
        if len(case.truth) > 1:
            raise ValueError(
                "Compound truth requires a multilabel calibration protocol"
            )
        target = (
            case.truth[0]
            if case.truth and case.truth[0] in probabilities
            else "unlisted"
        )
        rows.append(
            {
                "case": case.id,
                "probabilities": probabilities,
                "target": target,
                "input_sha256": hashlib.sha256(raw).hexdigest(),
                "case_sha256": hashlib.sha256(
                    json.dumps(asdict(case), sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--evaluate-draws", type=Path)
    args = parser.parse_args()
    if args.out.exists() or (args.plot and args.out.with_suffix("").exists()):
        parser.error("Output already exists")
    rows = load_rows(args.draws, "dev")
    report = audit(rows)
    if args.evaluate_draws:
        heldout = load_rows(args.evaluate_draws, "heldout")
        report["heldout"] = {
            "cases": [row["case"] for row in heldout],
            "before": metrics(heldout, 1),
            "after": metrics(heldout, report["temperature"]),
            "temperature_refitted": False,
        }
    report["source_sha256"] = {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in ["bench/prior_calibration.py", "core/agent/prior_calibration.py"]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    if args.plot:
        from bench.reliability_plot import plot_report

        destination = args.out.with_suffix("")
        destination.mkdir()
        plot_report(
            {
                "split": "dev, n=5",
                "methods": {
                    label: {"metrics": {"all": report[key]["reliability"]}}
                    for label, key in [
                        ("Raw priors", "before"),
                        ("Dev fit", "fitted_dev"),
                    ]
                },
            },
            destination,
        )
    print(
        json.dumps(
            {k: report[k] for k in ["temperature", "before", "fitted_dev"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
