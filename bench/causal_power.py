"""B10: simulate the implemented causal decision rule, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from core.interventions.stats import causal_check


def simulate(*, effect, noise_sd, seeds, trials, min_effect, random_seed=0):
    for name, value in (
        ("effect", effect),
        ("noise_sd", noise_sd),
        ("min_effect", min_effect),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if type(seeds) is not int or not 2 <= seeds <= 8:
        raise ValueError("seeds must be an integer from 2 to 8")
    if type(trials) is not int or not 1 <= trials <= 10000:
        raise ValueError("trials must be an integer from 1 to 10000")
    rng = random.Random(random_seed)
    counts = Counter()
    for _ in range(trials):
        control = [rng.gauss(0, noise_sd) for _ in range(seeds)]
        fix = [rng.gauss(effect, noise_sd) for _ in range(seeds)]
        result = causal_check(fix, control, min_effect=min_effect, noise_sd=noise_sd)
        counts[result.status] += 1
    successes = counts["confirmed"]
    p = successes / trials
    # Wilson interval describes Monte Carlo error, not model/domain uncertainty.
    z = 1.959963984540054
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return {
        "effect": effect,
        "noise_sd": noise_sd,
        "seeds_per_arm": seeds,
        "trials": trials,
        "random_seed": random_seed,
        "min_effect": min_effect,
        "counts": {s: counts[s] for s in ("confirmed", "rejected", "inconclusive")},
        "confirmation_rate": p,
        "mc_interval_95": [max(0, center - half), min(1, center + half)],
        "permutation_gate_active": math.comb(2 * seeds, seeds) * 0.05 >= 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-sd", type=float, default=0.019)
    parser.add_argument("--min-effect", type=float, default=0.06)
    parser.add_argument(
        "--effects", type=float, nargs="+", default=[0, 0.03, 0.06, 0.10]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists")
    rows = [
        simulate(
            effect=e,
            noise_sd=args.noise_sd,
            seeds=n,
            trials=args.trials,
            min_effect=args.min_effect,
            random_seed=args.random_seed,
        )
        for e in args.effects
        for n in args.seeds
    ]
    source = Path(__file__).resolve().parents[1] / "core/interventions/stats.py"
    report = {
        "version": 1,
        "assumptions": "Independent normal seed-level metrics; known, equal SD in both arms",
        "interval_scope": "Monte Carlo sampling error only",
        "stats_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "simulator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(
            f"effect={row['effect']:.3f} seeds={row['seeds_per_arm']}: confirmed={row['confirmation_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
