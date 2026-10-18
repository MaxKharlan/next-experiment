"""Fit recipe-probe likelihoods from dev cases with symmetric pseudocounts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml

from bench.cases import CASES_DIR, load_cases
from core.agent.context import InvestigationContext
from core.planner.outcome_model import SIGNATURES, load_signatures
from core.probes.level0.recipe import LEVEL0


def fit(cases, baseline, alpha=1.0):
    if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not cases or any(c.split != "dev" for c in cases):
        raise ValueError("Only a nonempty dev split may be fitted")
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs")
    jobs = [c.job_id for c in cases if c.job_id != "local"]
    if len(jobs) != len(set(jobs)):
        raise ValueError("Duplicate provider jobs would overweight observations")
    if any(len(c.truth) > 1 for c in cases):
        raise ValueError("Compound causes cannot identify single-cause likelihoods")
    known = {fault for rows in baseline.values() for fault in rows} - {"default"}
    if any(c.truth and c.truth[0] not in known for c in cases):
        raise ValueError("Unknown truth class")
    table = copy.deepcopy(baseline)
    observations = []
    for case in sorted(cases, key=lambda c: c.id):
        config = case.job_config()
        hp = config["hyperparameters"]
        for key in ("learning_rate", "n_epochs", "lora_r"):
            value = hp.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid {key} in {case.id}")
        points = config["checkpoints"]
        if len(points) < 2:
            raise ValueError(f"At least two checkpoints required for {case.id}")
        for point in points:
            for key in ("train_loss", "valid_loss"):
                value = point.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"Invalid {key} in {case.id}")
        if points[0]["train_loss"] == 0:
            raise ValueError("Initial training loss must be positive")
        ctx = InvestigationContext(case.id, "A", case.job_id, config)
        observations.append(
            {
                "case": case.id,
                "truth": case.truth,
                "outcomes": {p.name: p.run(ctx).outcome for p in LEVEL0},
            }
        )
    coverage = {}
    for probe in LEVEL0:
        if set(baseline[probe.name]["default"]) != set(probe.outcomes):
            raise ValueError(f"Outcome schema mismatch for {probe.name}")
        coverage[probe.name] = {}
        labels = sorted({c.truth[0] for c in cases if c.truth})
        for label in labels + ["healthy"]:
            selected = [
                r
                for r in observations
                if r["truth"] == ([] if label == "healthy" else [label])
            ]
            counts = Counter(r["outcomes"][probe.name] for r in selected)
            n = len(selected)
            probabilities = {
                o: (counts[o] + alpha) / (n + alpha * len(probe.outcomes))
                for o in probe.outcomes
            }
            coverage[probe.name][label] = {
                "n": n,
                "counts": {o: counts[o] for o in probe.outcomes},
                "smoothed": probabilities,
            }
            if label != "healthy":
                table[probe.name][label] = probabilities
    return table, {
        "version": 1,
        "fit_split": "dev",
        "alpha": alpha,
        "formula": "(count + alpha) / (n + alpha * number_of_outcomes)",
        "observations": observations,
        "coverage": coverage,
        "default_policy": "Unchanged: healthy cases do not represent all unlisted causes",
        "unobserved_policy": "Retain baseline rows; these remain uncalibrated",
        "active_probes": [p.name for p in LEVEL0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, default=CASES_DIR)
    parser.add_argument("--baseline", type=Path, default=SIGNATURES)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory already exists")
    cases = load_cases("dev", args.cases_root)
    table, report = fit(cases, load_signatures(args.baseline), args.alpha)
    report["baseline_sha256"] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    report["case_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((args.cases_root / "dev").glob("*.json"))
    }
    report["probe_source_sha256"] = hashlib.sha256(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("core/probes/level0/recipe.py")
        .read_bytes()
    ).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    output = yaml.safe_dump(table, sort_keys=True)
    report["fitted_sha256"] = hashlib.sha256(output.encode()).hexdigest()
    (args.out / "signatures.yaml").write_text(output)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Fitted {len(cases)} dev cases; saved {args.out}")


if __name__ == "__main__":
    main()
