"""B10: noise floor between seeds and seeds needed for a causal check, from sweep runs.

Reads <runs>/<config>-s<seed>/summary.json written by jobs/kaggle/sweep.py and
prints the mean and SD per config, the pooled noise floor, the minimum detectable
effect for 2 to 5 seeds, and a causal check of the healthy recipe against each
fault: does restoring the recipe beat keeping the fault?
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from core.interventions.stats import (
    causal_check,
    min_detectable_effect,
    noise_floor,
)

HIGHER_IS_BETTER = {
    "exact_match": True,
    "valid_mean_token_accuracy": True,
    "valid_loss": False,
}


def load(runs: Path, metric: str) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for summary in sorted(runs.glob("*/summary.json")):
        config = summary.parent.name.rsplit("-s", 1)[0]
        groups[config].append(json.loads(summary.read_text())["final"][metric])
    return dict(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument(
        "--metric", choices=sorted(HIGHER_IS_BETTER), default="exact_match"
    )
    parser.add_argument("--min-effect", type=float, help="default: two noise-floor SDs")
    args = parser.parse_args()

    groups = load(Path(args.runs), args.metric)
    floor = noise_floor(groups)
    min_effect = args.min_effect or 2 * floor.sd
    for config, values in groups.items():
        sd = statistics.stdev(values) if len(values) > 1 else float("nan")
        print(
            f"{config:30s} n={len(values)}  {statistics.fmean(values):.4f} ± {sd:.4f}"
        )
    print(
        f"noise floor SD {floor.sd:.4f} over {floor.runs} runs in {floor.groups} configs"
    )
    for seeds in range(2, 6):
        mde = min_detectable_effect(floor.sd, seeds)
        print(f"  {seeds} seeds per arm find an effect of {mde:.4f} with 80% power")
    print(
        f"causal check, healthy recipe against each fault, min effect {min_effect:.4f}"
    )
    for config, values in groups.items():
        if config == "healthy":
            continue
        r = causal_check(
            groups["healthy"],
            values,
            min_effect=min_effect,
            noise_sd=floor.sd,
            higher_is_better=HIGHER_IS_BETTER[args.metric],
        )
        print(
            f"  {config:30s} {r.status:12s} effect {r.effect:+.4f} "
            f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] p={r.p_value:.3f}"
        )


if __name__ == "__main__":
    main()
