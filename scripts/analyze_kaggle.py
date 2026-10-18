"""Analyze exported Kaggle results without training, model loading or API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.report.kaggle import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--min-effect", required=True, type=float)
    parser.add_argument("--noise-sd", type=float, default=None)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        report = analyze(
            args.results, min_effect=args.min_effect, noise_sd=args.noise_sd
        )
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.out:
            # Never overwrite an input, a previous report, or a symlink target.
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(text)
        print(text, end="")
    except (OSError, ValueError, TypeError, KeyError) as err:
        parser.exit(2, f"Invalid Kaggle results: {err}\n")


if __name__ == "__main__":
    main()
