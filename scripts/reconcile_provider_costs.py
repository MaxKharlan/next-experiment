"""Recompute or verify a frozen, redacted provider-cost reconciliation bundle."""

import argparse
import hashlib
import json
from pathlib import Path

from core.report.cost_reconciliation import reconcile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    report = reconcile(json.loads(raw))
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    if args.verify:
        if json.loads(args.out.read_text()) != report:
            raise SystemExit("Saved reconciliation does not match input")
    else:
        if args.input.resolve() == args.out.resolve():
            raise SystemExit("Output must not overwrite input")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "aggregate_training_usage_matches": report[
                    "aggregate_training_usage_matches"
                ],
                "invoice_reconciled": report["invoice_reconciled"],
                "display_total_delta_usd": report["display_total_delta_usd"],
                "discrepancies": len(report["discrepancies"]),
            }
        )
    )
    if report["discrepancies"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
