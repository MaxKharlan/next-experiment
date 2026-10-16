"""B18: a held-out set for comparing the base, failed and fixed models.

Samples BANKING77 test queries that are not in the validation file, so the
comparison does not reuse the examples the causal checks were scored on. A
stand-in until track A's split with a held-out part (A1).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scripts.measure_post_training import load_split, to_chat, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid", default="artifacts/b6/valid.jsonl")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="artifacts/b18/heldout.jsonl")
    args = parser.parse_args()

    lines = Path(args.valid).read_text(encoding="utf-8").splitlines()
    seen = {
        m["content"]
        for line in lines
        if line.strip()
        for m in json.loads(line)["messages"]
        if m["role"] == "user"
    }
    pool = [row for row in load_split("test") if row[0] not in seen]
    rows = random.Random(args.seed).sample(pool, args.n)
    write_jsonl([to_chat(*row) for row in rows], Path(args.out))
    print(f"held-out {len(rows)} queries in {args.out}, none of them in {args.valid}")


if __name__ == "__main__":
    main()
