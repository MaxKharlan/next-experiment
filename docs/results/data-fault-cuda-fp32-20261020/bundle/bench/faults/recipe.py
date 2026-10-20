"""Faults 6–8: training recipe changes that make a LoRA fine-tune underperform.

Each fault takes healthy hyperparameters in Token Factory names and a strength and
returns a changed copy. The strengths in STRENGTHS are the ones from the project plan.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

# The recipe of the B6 jobs. Token Factory's own defaults differ: learning_rate 1e-5,
# 3 epochs, lora_r 8, lora_alpha 8, dropout 0, packing on (post-training docs)
HEALTHY_HP: dict[str, Any] = {
    "n_epochs": 1,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.0,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "context_length": 8192,
}

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


def grid(
    seeds: list[int], faults: list[str] | None = None
) -> Iterator[tuple[str, str | None, float | None, int]]:
    """(run name, fault, strength, seed): a healthy run, then each fault and strength.

    Seeds are the outer loop, so a partial sweep has every fault for the first seeds.
    """
    for seed in seeds:
        yield "healthy", None, None, seed
        for fault in STRENGTHS if faults is None else faults:
            for level, strength in STRENGTHS[fault].items():
                yield f"{fault}-{level}", fault, strength, seed
