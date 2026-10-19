"""Resolve a portable InvestigationCase and audit it without model calls."""

import argparse
import json
from pathlib import Path

from bench.tasks.structured import load_case
from core.eval.audit_case import audit_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--request-only", action="store_true")
    args = parser.parse_args()
    case = load_case(args.case)
    print(
        json.dumps(
            case.model_dump() if args.request_only else audit_case(case), indent=2
        )
    )


if __name__ == "__main__":
    main()
