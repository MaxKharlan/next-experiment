"""Faults 6–8: training recipe changes that make a LoRA fine-tune underperform.

Each fault takes healthy hyperparameters in Token Factory names and a strength and
returns a changed copy. The strengths in STRENGTHS are the ones from the project plan.
"""

from __future__ import annotations

from typing import Any

# fault class from core.agent.schemas.FaultClass -> named strengths
STRENGTHS: dict[str, dict[str, float]] = {
    "lr_too_low": {"mild": 0.1, "strong": 0.03},  # learning rate multiplier
    "too_many_epochs": {"mild": 3, "strong": 6},  # epochs multiplier
    "lora_rank_too_small": {"mild": 4, "strong": 2},  # LoRA rank instead of 16
}


def inject(hp: dict[str, Any], fault: str, strength: float) -> dict[str, Any]:
    """Return a copy of hp with the fault applied at the given strength."""
    out = dict(hp)
    if fault == "lr_too_low":
        out["learning_rate"] = hp["learning_rate"] * strength
    elif fault == "too_many_epochs":
        out["n_epochs"] = round(hp["n_epochs"] * strength)
    elif fault == "lora_rank_too_small":
        # lora_alpha stays as it was, like a user who only lowers the rank
        out["lora_r"] = int(strength)
    else:
        raise ValueError(f"unknown recipe fault {fault!r}")
    return out
