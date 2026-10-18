"""Heuristic belief updates after a causal check.

These likelihood ratios are policy assumptions, not calibrated probabilities.
A beneficial intervention can also compensate for a different underlying cause.
"""

import math
from collections.abc import Mapping

MODEL = "causal-odds-heuristic-v1"
ODDS_MULTIPLIERS = {
    "confirmed": 3.0,
    "rejected": 1.0 / 3.0,
    "inconclusive": 1.0,
}


def update_causal(
    probs: Mapping[str, float],
    classes: Mapping[str, str],
    hypothesis_id: str,
    status: str,
) -> dict[str, float]:
    """Update the whole tested fault class and preserve other class ratios."""
    multiplier = ODDS_MULTIPLIERS[status]
    tested_class = classes[hypothesis_id]

    if not probs or hypothesis_id not in probs:
        raise ValueError("The tested hypothesis must have a prior probability")
    if any(not math.isfinite(p) or p < 0 for p in probs.values()):
        raise ValueError("Probabilities must be finite and nonnegative")
    if sum(probs.values()) <= 0:
        raise ValueError("Probability mass must be positive")

    weights = {
        h: p * (multiplier if classes.get(h) == tested_class else 1.0)
        for h, p in probs.items()
    }
    total = sum(weights.values())
    return {h: p / total for h, p in weights.items()}
