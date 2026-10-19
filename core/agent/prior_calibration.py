"""Temperature scaling over proposed classes plus explicit unlisted mass."""

from __future__ import annotations

import math

EPSILON = 1e-9


def distribution(hypotheses):
    result = {}
    for h in hypotheses:
        if h.fault_class == "unlisted":
            raise ValueError("unlisted is reserved for residual mass")
        if not math.isfinite(h.prior) or not 0 <= h.prior <= 1:
            raise ValueError("Invalid prior")
        result[h.fault_class] = result.get(h.fault_class, 0.0) + h.prior
    total = sum(result.values())
    if total > 1 + 1e-9:
        raise ValueError("Prior mass exceeds one")
    result["unlisted"] = max(0.0, 1 - total)
    return result


def scale_distribution(probabilities, temperature):
    if (
        isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Temperature must be finite and positive")
    if (
        not probabilities
        or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
        or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-8)
    ):
        raise ValueError("Expected a probability distribution")
    if temperature == 1:
        return dict(probabilities)
    logits = {
        k: math.log(max(EPSILON, p)) / temperature for k, p in probabilities.items()
    }
    peak = max(logits.values())
    weights = {k: math.exp(v - peak) for k, v in logits.items()}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def calibrate(hypotheses, temperature):
    before = distribution(hypotheses)
    after = scale_distribution(before, temperature)
    # Keep duplicate locations in their original within-class proportions.
    counts = {k: sum(h.fault_class == k for h in hypotheses) for k in before}
    adjusted = [
        h.model_copy(
            update={
                "prior": after[h.fault_class]
                * (
                    h.prior / before[h.fault_class]
                    if before[h.fault_class]
                    else 1 / counts[h.fault_class]
                )
            }
        )
        for h in hypotheses
    ]
    return adjusted, {
        "kind": "prior_calibration",
        "temperature": temperature,
        "before": before,
        "after": after,
        "residual_meaning": "none of the proposed classes; not a healthy probability",
    }
