"""Verify saved historical-Trainer evidence without importing ML packages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.reproduce_accumulation_remainder import (
    CASES,
    RATES,
    TOLERANCE,
    analytic,
    sha,
    validate_result,
)


def check_manifest(root, expected):
    manifest = json.loads((root / "manifest.json").read_text())
    if set(manifest) != expected or {p.name for p in root.iterdir()} != {
        *expected,
        "manifest.json",
    }:
        raise ValueError("Manifest coverage differs")
    for name, digest in manifest.items():
        if sha(root / name) != digest:
            raise ValueError(f"Evidence changed: {name}")
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        sha(root / "protocol.json") != (root / "protocol.sha256").read_text().strip()
        or sha(root / "runner-source.txt") != protocol["script_sha256"]
    ):
        raise ValueError("Protocol/source binding differs")
    return protocol


def summarize(root):
    protocol = check_manifest(
        root,
        {
            "protocol.json",
            "protocol.sha256",
            "runner-source.txt",
            "failed-results.json",
            "requirements.txt",
        },
    )
    if (
        protocol["id"] != "transformers-38837-cpu-mechanism-v2"
        or protocol["cases"] != [{"examples": n, "accumulation": k} for n, k in CASES]
        or protocol["learning_rates"] != list(RATES)
        or protocol["absolute_tolerance"] != TOLERANCE
    ):
        raise ValueError("Changed experiment")
    result = json.loads((root / "failed-results.json").read_text())
    if result["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Result binding differs")
    if (
        result["status"] != "reproduction_not_achieved"
        or result["downstream_quality_measured"] is not False
    ):
        raise ValueError("Failure must remain visible")
    # The original validator stops at the first mismatch. Check the full design
    # here so rows after that mismatch cannot silently change the comparison.
    if [
        (row["examples"], row["accumulation"], row["learning_rate"])
        for row in result["cases"]
    ] != [(n, k, rate) for n, k in CASES for rate in RATES]:
        raise ValueError("Changed design")
    # The prespecified pass criterion must continue to fail for the saved reason.
    try:
        validate_result({**result, "status": "mechanism_reproduced"})
    except ValueError as error:
        if str(error) != result["error"]:
            raise ValueError("Recorded failure differs") from error
    else:
        raise ValueError("Expected a failed experiment")
    rows = []
    for row in result["cases"]:
        n, k, rate = row["examples"], row["accumulation"], row["learning_rate"]
        comparisons = {}
        for arm in ("original", "corrected"):
            expected = analytic(n, k, rate, buggy=arm == "original")
            observed = row[arm]
            if len(expected) != len(observed):
                raise ValueError("Unexpected update count")
            errors = []
            for actual, reference in zip(observed, expected):
                for key in ("gradient", "parameters"):
                    if len(actual[key]) != 3 or any(
                        isinstance(x, bool)
                        or not isinstance(x, (float, int))
                        or not math.isfinite(x)
                        for x in actual[key]
                    ):
                        raise ValueError("Invalid measured values")
                    errors.extend(
                        abs(x - y) for x, y in zip(actual[key], reference[key])
                    )
            comparisons[arm] = {
                "max_absolute_error": max(errors),
                "matches_prespecified_reference": max(errors) <= TOLERANCE,
                "final_gradient": observed[-1]["gradient"],
                "final_parameters": observed[-1]["parameters"],
            }
        rows.append(
            {
                "examples": n,
                "accumulation": k,
                "learning_rate": rate,
                "comparisons": comparisons,
            }
        )
    return {
        "status": "stopped",
        "protocol_sha256": result["protocol_sha256"],
        "input_manifest_sha256": sha(root / "manifest.json"),
        "prespecified_failure": result["error"],
        "trainer_runs": 12,
        "matching_cases": sum(
            all(
                arm["matches_prespecified_reference"]
                for arm in row["comparisons"].values()
            )
            for row in rows
        ),
        "cases": rows,
        "downstream_quality_measured": False,
        "service_spend_usd": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    report = json.dumps(summarize(args.root), indent=2, sort_keys=True) + "\n"
    if args.verify:
        if args.out.read_text() != report:
            raise ValueError("Report differs from saved measurements")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as file:
            file.write(report)
    print("Verified stopped experiment: 12 Trainer runs, no quality claim")


if __name__ == "__main__":
    main()
