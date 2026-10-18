"""Fit a binary confidence temperature on dev and evaluate it without refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from bench.calibration import reliability

EPSILON = 1e-6
BOUNDS = (0.05, 20.0)
SCOPE = "top-candidate correctness"


def scale(probability, temperature):
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Temperature must be finite and positive")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("Probability must be finite and in [0, 1]")
    p = min(1 - EPSILON, max(EPSILON, probability))
    logit = (math.log(p) - math.log1p(-p)) / temperature
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exp = math.exp(logit)
    return exp / (1 + exp)


def nll(samples, temperature):
    losses = []
    for p, y in samples:
        p = min(1 - 1e-15, max(1e-15, scale(p, temperature)))
        losses.append(-math.log(p) if y else -math.log1p(-p))
    return sum(losses) / len(losses)


def fit_temperature(samples):
    reliability(samples)
    if len({correct for _, correct in samples}) != 2:
        raise ValueError("Fitting needs both correct and incorrect dev candidates")
    # NLL is convex in inverse temperature; its derivative is monotonic.
    logits = [
        math.log(min(1 - EPSILON, max(EPSILON, p)))
        - math.log1p(-min(1 - EPSILON, max(EPSILON, p)))
        for p, _ in samples
    ]
    low, high = 1 / BOUNDS[1], 1 / BOUNDS[0]
    for _ in range(80):
        middle = (low + high) / 2
        derivative = sum(
            z * (scale(p, 1 / middle) - y)
            for z, (p, y) in zip(logits, samples, strict=True)
        )
        if derivative > 0:
            high = middle
        else:
            low = middle
    candidates = [1.0, *BOUNDS, 1 / ((low + high) / 2)]
    return min(candidates, key=lambda t: (nll(samples, t), abs(math.log(t))))


def checked_rows(audit):
    if audit.get("version") != 1 or audit.get("scope") != SCOPE:
        raise ValueError("Expected a version 1 top-candidate confidence audit")
    methods = audit.get("methods")
    if not methods:
        raise ValueError("Audit contains no methods")
    expected = None
    for method, entry in methods.items():
        rows = entry["cases"]
        ids = [row["case"] for row in rows]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError(f"Empty or duplicate cases for {method}")
        if expected is not None and set(ids) != expected:
            raise ValueError("All methods must cover the same cases")
        expected = set(ids)
        reliability([(row["confidence"], row["correct"]) for row in rows])
    return methods


def fit(audit):
    if audit.get("split") != "dev":
        raise ValueError("Fit temperatures on dev only")
    output = {}
    for method, entry in checked_rows(audit).items():
        samples = [(r["confidence"], r["correct"]) for r in entry["cases"]]
        temperature = fit_temperature(samples)
        output[method] = {
            "temperature": temperature,
            "fit_case_ids": [r["case"] for r in entry["cases"]],
            "n": len(samples),
            "dev_nll_before": nll(samples, 1),
            "dev_nll_after": nll(samples, temperature),
            "at_search_boundary": any(math.isclose(temperature, b) for b in BOUNDS),
        }
    return {
        "version": 1,
        "scope": SCOPE,
        "fit_split": "dev",
        "epsilon": EPSILON,
        "temperature_bounds": BOUNDS,
        "objective": "binary negative log likelihood",
        "methods": output,
    }


def evaluate(audit, fitted):
    if audit.get("split") != "heldout":
        raise ValueError(
            "Evaluate on a disjoint heldout split; dev fit losses are in the model"
        )
    if (
        fitted.get("version") != 1
        or fitted.get("scope") != SCOPE
        or fitted.get("fit_split") != "dev"
        or fitted.get("epsilon") != EPSILON
    ):
        raise ValueError("Unsupported fitted temperature artifact")
    methods = checked_rows(audit)
    if set(methods) != set(fitted["methods"]):
        raise ValueError("Evaluation methods must match fitted methods")
    output = {}
    for method, entry in methods.items():
        model = fitted["methods"][method]
        rows = entry["cases"]
        if set(model["fit_case_ids"]) & {r["case"] for r in rows}:
            raise ValueError("Fit and evaluation case IDs overlap")
        temperature = model["temperature"]
        before = [(r["confidence"], r["correct"]) for r in rows]
        after = [(scale(p, temperature), y) for p, y in before]
        output[method] = {
            "temperature": temperature,
            "before": reliability(before),
            "after": reliability(after),
            "nll_before": nll(before, 1),
            "nll_after": nll(before, temperature),
            "cases": [
                {**r, "calibrated_confidence": scale(r["confidence"], temperature)}
                for r in rows
            ],
        }
    return {
        "version": 1,
        "scope": SCOPE,
        "split": "heldout",
        "refitted": False,
        "methods": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fit", "evaluate"])
    parser.add_argument("audit", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists")
    if (args.action == "evaluate") != (args.model is not None):
        parser.error("--model is required only for evaluate")
    raw = args.audit.read_bytes()
    audit = json.loads(raw)
    if args.action == "fit":
        result = fit(audit)
    else:
        model_raw = args.model.read_bytes()
        result = evaluate(audit, json.loads(model_raw))
        result["model_sha256"] = hashlib.sha256(model_raw).hexdigest()
    result["audit_sha256"] = hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"Saved temperature {args.action} to {args.out}")


if __name__ == "__main__":
    main()
