"""Reproduce the BANKING77 epoch sensitivity grid (paid execution is opt-in)."""

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.faults.recipe import HEALTHY_HP
from core.connectors.token_factory import TokenFactory
from scripts.tf_fault_sweep import plan_runs, run_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--parallel-seeds", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    rows = args.train.read_text().splitlines()
    if len(rows) != 500:
        raise ValueError("This protocol requires 500 training examples")
    manifest = {
        "model": "Qwen/Qwen3-0.6B",
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "valid_sha256": hashlib.sha256(args.valid.read_bytes()).hexdigest(),
        "seeds": [0, 1, 2],
        "epochs": [3, 9, 18],
        "hyperparameters": {**HEALTHY_HP, "n_epochs": 3},
        "scope": "descriptive validation study; local exact-match evaluation",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Resume data or protocol changed")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.execute:
        return

    def run_seed(seed):
        tf = TokenFactory.from_settings()
        tf.client = tf.client.with_options(timeout=60, max_retries=0)
        data = list(rows)
        random.Random(seed).shuffle(data)
        train = args.out / f"train-s{seed}.jsonl"
        train.write_text("\n".join(data) + "\n")
        runs = plan_runs(
            manifest["model"], manifest["hyperparameters"], [seed], ["too_many_epochs"]
        )
        run_grid(
            tf, runs, args.out / f"s{seed}", str(train), str(args.valid), parallel=3
        )

    with ThreadPoolExecutor(max_workers=args.parallel_seeds) as pool:
        list(pool.map(run_seed, manifest["seeds"]))


if __name__ == "__main__":
    main()
