"""B16: what the loop does next: run a probe, test the leader, or stop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

Decision = Literal["probe", "intervene", "stop"]

# a probe that gains less than this many bits is not worth running
MIN_GAIN_BITS = 0.01
# with no informative probe left, a leader below this does not get a causal check;
# calibrated on the dev set, where healthy jobs otherwise paid for checks of
# hypotheses at 0.04 to 0.24
MIN_CHECK_PROB = 0.25


def next_action(
    probs: Mapping[str, float],
    *,
    threshold: float,
    gain_bits: float,
    budget_left_usd: float,
    intervention_cost_usd: float,
    testable: set[str],
    min_prob: float = MIN_CHECK_PROB,
) -> tuple[Decision, str]:
    """testable holds the hypotheses that have an intervention and were not tested yet."""
    leader = max((h for h in probs if h in testable), key=probs.get, default=None)
    if leader is None:
        return "stop", "no hypothesis left to test"
    if gain_bits < MIN_GAIN_BITS and probs[leader] < min(threshold, min_prob):
        return "stop", (
            f"no probe left and the likeliest testable hypothesis, {leader} at "
            f"{probs[leader]:.2f}, is too unlikely to pay for a causal check"
        )
    if probs[leader] >= threshold or gain_bits < MIN_GAIN_BITS:
        if budget_left_usd < intervention_cost_usd:
            return "stop", "the budget does not cover a causal check"
        return "intervene", f"{leader} at {probs[leader]:.2f}"
    return "probe", f"leader {leader} at {probs[leader]:.2f} is below {threshold}"
