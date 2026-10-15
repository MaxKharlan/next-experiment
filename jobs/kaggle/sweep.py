"""B9: a grid of LoRA runs, healthy and with faults 6–8 at each strength.

Each run goes to <out>/<run>-s<seed>/ with the files that train_lora.py writes.
Runs with a summary.json are skipped, so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.faults.recipe import STRENGTHS, grid
from jobs.kaggle.train_lora import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--hp", default="{}", help="JSON overrides of the healthy run")
    parser.add_argument(
        "--faults", nargs="*", choices=sorted(STRENGTHS), help="default: all"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--lr-schedule", choices=["linear", "constant"], default="linear"
    )
    parser.add_argument("--generate", action="store_true", help="exact match on valid")
    args = parser.parse_args()

    for name, fault, strength, seed in grid(args.seeds, args.faults):
        run_dir = Path(args.out) / f"{name}-s{seed}"
        if (run_dir / "summary.json").exists():
            print(f"{run_dir.name}: done before, skipped", flush=True)
            continue
        s = train(
            args.train,
            args.valid,
            run_dir,
            model_id=args.model,
            hp=json.loads(args.hp),
            fault=fault,
            strength=strength,
            seed=seed,
            lr_schedule=args.lr_schedule,
            generate=args.generate,
        )
        print(f"{run_dir.name}: {s['wall_seconds']} s, {s['final']}", flush=True)


if __name__ == "__main__":
    main()
