"""Inspect explicit train/validation files for lexical overlap candidates."""

import argparse
import json
from pathlib import Path

from core.probes.level0.overlap import DEFAULT_MAX_PAIRS, audit_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    args = parser.parse_args()
    result = audit_files(args.train, args.validation, max_pairs=args.max_pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as file:
        file.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"{len(result['candidate_pairs'])} candidate pairs; review required")


if __name__ == "__main__":
    main()
