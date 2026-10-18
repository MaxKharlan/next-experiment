"""Audit top-candidate confidence from saved diagnoses without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from bench.cases import load_cases


def reliability(samples, bins=5):
    """Binary Brier and ECE for the claim that the selected cause is correct."""
    if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 20:
        raise ValueError("bins must be an integer between 1 and 20")
    if not samples:
        raise ValueError("At least one sample is required")
    groups = [[] for _ in range(bins)]
    for confidence, correct in samples:
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or type(correct) is not bool
        ):
            raise ValueError(
                "Expected finite confidence in [0, 1] and boolean correctness"
            )
        groups[min(int(confidence * bins), bins - 1)].append((confidence, correct))
    table = []
    for index, group in enumerate(groups):
        mean_confidence = sum(p for p, _ in group) / len(group) if group else None
        accuracy = sum(y for _, y in group) / len(group) if group else None
        table.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "upper_inclusive": index == bins - 1,
                "count": len(group),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "n": len(samples),
        "accuracy": sum(y for _, y in samples) / len(samples),
        "mean_confidence": sum(p for p, _ in samples) / len(samples),
        "binary_brier": sum((p - y) ** 2 for p, y in samples) / len(samples),
        "ece": sum(
            row["count"] * abs(row["mean_confidence"] - row["accuracy"])
            for row in table
            if row["count"]
        )
        / len(samples),
        "bins": table,
    }


def selected_cause(ranking):
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("A nonempty ranking is required")
    seen = set()
    for entry in ranking:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("Ranking entries must be [label, probability]")
        label, probability = entry
        if not isinstance(label, str) or not label or label in seen:
            raise ValueError("Labels must be nonempty and unique")
        seen.add(label)
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("Probabilities must be finite and in [0, 1]")
    if sum(p for _, p in ranking) > 1 + 1e-6:
        raise ValueError("Ranking probability mass exceeds one")
    # Diagnosis may promote a confirmed cause ahead of a higher probability.
    # Select by confidence here; ties use the label for deterministic results.
    return min(ranking, key=lambda entry: (-entry[1], entry[0]))


def analyze(records, cases, bins=5):
    if not cases:
        raise ValueError("No benchmark cases found")
    known = {case.id: case for case in cases}
    groups = {}
    seen = set()
    for record in records:
        case_id = record["case"]
        method = record.get("variant", record.get("method"))
        if not isinstance(method, str) or not method:
            raise ValueError("A variant or method is required")
        if case_id not in known:
            raise ValueError(f"Case outside the selected split: {case_id}")
        if (method, case_id) in seen:
            raise ValueError(f"Duplicate method/case: {method}/{case_id}")
        seen.add((method, case_id))
        label, confidence = selected_cause(record["ranking"])
        case = known[case_id]
        # No residual mass is reinterpreted as a probability of being healthy.
        correct = label in case.truth
        groups.setdefault(method, []).append(
            {
                "case": case_id,
                "candidate": label,
                "confidence": confidence,
                "correct": correct,
                "healthy": case.healthy,
                "truth": case.truth,
            }
        )
    if not groups:
        raise ValueError("No predictions found")
    output = {}
    for method, rows in groups.items():
        missing = set(known) - {row["case"] for row in rows}
        if missing:
            raise ValueError(f"Incomplete split for {method}: {sorted(missing)}")
        strata = {}
        for name, selected in (
            ("all", rows),
            ("faults", [r for r in rows if not r["healthy"]]),
            ("healthy", [r for r in rows if r["healthy"]]),
        ):
            strata[name] = (
                reliability([(r["confidence"], r["correct"]) for r in selected], bins)
                if selected
                else None
            )
        output[method] = {"metrics": strata, "cases": rows}
    return {
        "version": 1,
        "scope": "top-candidate correctness",
        "selection": "maximum reported probability; label order breaks ties",
        "probabilities_renormalized": False,
        "calibration_fitted": False,
        "methods": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="JSON array or JSONL with case, variant/method, ranking",
    )
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists")
    raw = args.input.read_bytes()
    source = raw.decode()
    records = (
        json.loads(source)
        if source.lstrip().startswith("[")
        else [json.loads(line) for line in source.splitlines() if line.strip()]
    )
    cases = load_cases(args.split)
    result = analyze(records, cases, args.bins)
    result.update(
        {
            "split": args.split,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "truth_sha256": hashlib.sha256(
                json.dumps({c.id: c.truth for c in cases}, sort_keys=True).encode()
            ).hexdigest(),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"Saved confidence audit to {args.out}")


if __name__ == "__main__":
    main()
