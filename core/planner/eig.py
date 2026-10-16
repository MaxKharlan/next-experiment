"""B16: Bayes updates and the probe with the highest expected information gain per dollar.

score(p) = (H(h) - sum_o P(o | p) H(h | o, p)) / cost(p), with
P(o | p) = sum_h P(h) P(o | h, p); see section 07 of the project doc.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from core.planner.outcome_model import likelihood, outcomes

# free checks still need a cost to divide by; they rank by information alone
MIN_COST_USD = 1e-4


def entropy(probs: Mapping[str, float]) -> float:
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)


def update(
    probs: Mapping[str, float], classes: Mapping[str, str], probe: str, outcome: str
) -> dict[str, float]:
    """P(h | o) ∝ P(h) P(o | h); classes maps a hypothesis id to its fault class."""
    raw = {h: p * likelihood(probe, classes[h], outcome) for h, p in probs.items()}
    total = sum(raw.values())
    if total == 0:
        return dict(probs)
    return {h: v / total for h, v in raw.items()}


def expected_gain(
    probs: Mapping[str, float], classes: Mapping[str, str], probe: str
) -> float:
    after = 0.0
    for o in outcomes(probe):
        p_o = sum(p * likelihood(probe, classes[h], o) for h, p in probs.items())
        if p_o > 0:
            after += p_o * entropy(update(probs, classes, probe, o))
    return entropy(probs) - after


def best_probe(
    probs: Mapping[str, float],
    classes: Mapping[str, str],
    costs: Mapping[str, float],
) -> tuple[str | None, float]:
    """The probe with the highest gain per dollar among costs, or None if none helps."""
    best, best_score = None, 0.0
    for probe, cost in costs.items():
        score = expected_gain(probs, classes, probe) / max(cost, MIN_COST_USD)
        if score > best_score + 1e-12:
            best, best_score = probe, score
    return best, best_score
