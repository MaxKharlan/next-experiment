"""B5: train the reference of a task on Token Factory and save its metrics.

For each base model four healthy recipes train on the task's training file (see
bench/reference.py). The one with the best validation exact match is the reference,
and its held-out exact match goes to bench/references/<task>.json, where the
benchmark reads the bar for fix success. The script prints the price and asks
before it starts any job. Defaults are BANKING77 with the files of the fault-6 case
and the held-out set from scripts/make_heldout.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.reference import GRID, ReferenceRun, grid_cost
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TokenFactory
from scripts.retrain_tf import approved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="banking77")
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3-0.6B"])
    parser.add_argument("--train", type=Path, default=Path("artifacts/b6/train.jsonl"))
    parser.add_argument("--valid", type=Path, default=Path("artifacts/b6/valid.jsonl"))
    parser.add_argument(
        "--heldout", type=Path, default=Path("artifacts/b18/heldout.jsonl")
    )
    # chat template tokens per BANKING77 example for Qwen3 (docs/b6-post-training.md)
    parser.add_argument("--tokens-per-example", type=float, default=57)
    parser.add_argument("--yes", action="store_true", help="approve without asking")
    args = parser.parse_args()

    if not args.heldout.exists():
        raise SystemExit(f"no held-out set at {args.heldout}: run scripts.make_heldout")
    lines = args.train.read_text(encoding="utf-8").splitlines()
    examples = sum(1 for line in lines if line.strip())
    cost = sum(grid_cost(m, examples, args.tokens_per_example) for m in args.models)
    print(
        f"References for {args.task}: {len(GRID)} jobs per model "
        f"({', '.join(GRID)}) on {examples} examples for {', '.join(args.models)}, "
        f"about ${cost:.3f} in total"
    )
    if not args.yes:
        answer = input("Start the jobs? [y/N] ")
        if not approved(answer):
            print(f"Not started: the answer was {answer!r}, not y.")
            return

    from jobs.eval_local.exact_match import evaluate_adapter

    tf, storage = TokenFactory.from_settings(), LocalStorage("artifacts")
    out = Path(f"bench/references/{args.task}.json")
    saved = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for model in args.models:
        result = ReferenceRun(
            tf=tf,
            storage=storage,
            model=model,
            train_path=args.train,
            valid_path=args.valid,
            heldout_path=args.heldout,
            score=evaluate_adapter,
            prefix=f"bench/references/{args.task}/{model.split('/')[-1]}",
        )()
        saved[model] = {"train_examples": examples, **result}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        print(
            f"{model}: reference {result['recipe']}, validation "
            f"{result['valid_exact_match']:.3f}, held-out "
            f"{result['heldout_exact_match']:.3f} ({result['job_id']})"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
