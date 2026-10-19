"""B10 and B11: noise floor between seeds, power, and the causal check verdict.

A check compares the targeted fix with its specified control across seeds.
The legacy status `confirmed` means the intervention meets the decision rule.
This calculation does not establish equal resources, unique cause or full recovery.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

Z = statistics.NormalDist().inv_cdf
# two-sided 95% quantiles of Student's t for 1..30 degrees of freedom
T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


@dataclass(frozen=True)
class NoiseFloor:
    sd: float  # pooled standard deviation of the metric between seeds
    runs: int
    groups: int


def noise_floor(groups: Mapping[str, Sequence[float]]) -> NoiseFloor:
    """Pooled between-seed SD over configs; each group is one config over seeds."""
    used = {k: v for k, v in groups.items() if len(v) >= 2}
    if not used:
        raise ValueError("need at least one config with two or more seeds")
    dof = sum(len(v) - 1 for v in used.values())
    ss = sum(statistics.variance(v) * (len(v) - 1) for v in used.values())
    return NoiseFloor(
        sd=math.sqrt(ss / dof), runs=sum(map(len, used.values())), groups=len(used)
    )


def seeds_for_power(
    effect: float, sd: float, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Seeds per arm for a one-sided two-sample test to find the effect."""
    n = 2 * ((Z(1 - alpha) + Z(power)) * sd / effect) ** 2
    return max(2, math.ceil(n))


def min_detectable_effect(
    sd: float, seeds: int, power: float = 0.8, alpha: float = 0.05
) -> float:
    return (Z(1 - alpha) + Z(power)) * sd * math.sqrt(2 / seeds)


def permutation_p(fix: Sequence[float], control: Sequence[float]) -> float:
    """One-sided p that the fix mean exceeds the control mean by chance."""
    if not fix or not control:
        raise ValueError("Permutation test requires two nonempty arms")
    if not all(math.isfinite(x) for x in (*fix, *control)):
        raise ValueError("Permutation values must be finite")
    pooled, n = list(fix) + list(control), len(fix)
    observed = statistics.fmean(fix) - statistics.fmean(control)
    total = len(pooled)
    sampled = math.comb(total, n) > 20_000
    if sampled:
        rng = random.Random(0)
        splits = (tuple(rng.sample(range(total), n)) for _ in range(20_000))
        count = 20_000
    else:
        splits = combinations(range(total), n)
        count = math.comb(total, n)
    hits = 0
    for idx in splits:
        chosen = set(idx)
        a = [pooled[i] for i in idx]
        b = [pooled[i] for i in range(total) if i not in chosen]
        hits += statistics.fmean(a) - statistics.fmean(b) >= observed - 1e-12
    # Include the observed assignment in the Monte Carlo estimate.
    return (hits + 1) / (count + 1) if sampled else hits / count


@dataclass(frozen=True)
class CausalResult:
    status: Literal["confirmed", "rejected", "inconclusive"]
    effect: float  # fix minus control, positive when the fix helps
    ci_low: float
    ci_high: float
    p_value: float
    min_effect: float
    reason: str


def causal_check(
    fix: Sequence[float],
    control: Sequence[float],
    *,
    min_effect: float,
    noise_sd: float | None = None,
    higher_is_better: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """Verdict on a fix versus its specified control, one value per seed and arm.

    min_effect is the smallest useful improvement, usually a few
    noise-floor SDs. With noise_sd from noise_floor() the interval uses it; without,
    it uses the pooled SD of these runs and Student's t.
    confirmed: effect >= min_effect, the interval is above zero and p <= alpha.
    The permutation test is skipped when it cannot reach alpha at all, as with
    2 seeds per arm (smallest p 1/6); the interval then decides alone.
    rejected: the whole interval is below min_effect.
    inconclusive: anything else; more seeds may decide it.
    """
    if len(fix) < 2 or len(control) < 2:
        raise ValueError("need at least two seeds per arm")
    sign = 1 if higher_is_better else -1
    fix = [sign * x for x in fix]
    control = [sign * x for x in control]
    effect = statistics.fmean(fix) - statistics.fmean(control)
    scale = math.sqrt(1 / len(fix) + 1 / len(control))
    if noise_sd is not None:
        half = Z(0.975) * noise_sd * scale
    else:
        dof = len(fix) + len(control) - 2
        pooled = noise_floor({"fix": fix, "control": control}).sd
        half = (T975[dof - 1] if dof <= 30 else Z(0.975)) * pooled * scale
    low, high = effect - half, effect + half
    p = permutation_p(fix, control)
    can_reach_alpha = math.comb(len(fix) + len(control), len(fix)) * alpha >= 1

    if effect >= min_effect and low > 0 and (p <= alpha or not can_reach_alpha):
        status, reason = "confirmed", "the fix beats the control beyond the noise"
    elif high < min_effect:
        status, reason = "rejected", "the fix gains less than the minimum effect"
    else:
        status, reason = "inconclusive", "the interval spans the minimum effect"
    return CausalResult(status, effect, low, high, p, min_effect, reason)
