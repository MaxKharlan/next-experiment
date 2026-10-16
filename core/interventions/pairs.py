"""B8: a targeted fix and a same-cost control for a hypothesis.

make_pair() turns a hypothesis about the failed job into two mini-job recipes. The
target changes only the factor the hypothesis blames. The control leaves that factor
alone and trains at least as many tokens as the target, so a win of the target can't
come from more training. It is the same cost for faults 6 and 8. For fault 7 the
target is cheaper: the control is the failed recipe itself, because any equal-cost
control there would also cut the number of distinct examples. Builders for faults
6–8 are here; data faults (A7) add theirs to BUILDERS.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from core.agent.context import InvestigationContext
from core.agent.schemas import Hypothesis


@dataclass
class Intervention:
    hypothesis_id: str
    kind: Literal["target", "control"]
    # hyperparameter changes against the failed job; "data_fraction" is the share
    # of training examples the mini-job uses
    patch: dict[str, Any]
    description: str
    ft_job_ids: list[str] = field(default_factory=list)
    metrics: list[float] = field(default_factory=list)  # one value per seed

    @property
    def data_fraction(self) -> float:
        return self.patch.get("data_fraction", 1.0)

    def hyperparameters(self, base: dict[str, Any]) -> dict[str, Any]:
        changes = {k: v for k, v in self.patch.items() if k != "data_fraction"}
        return {**base, **changes}

    def relative_cost(self, base: dict[str, Any]) -> float:
        """Trained tokens relative to one epoch over the whole training file."""
        return self.data_fraction * self.hyperparameters(base)["n_epochs"]


# hyperparameters of the failed job -> (target patch, its description,
# control patch, its description)
Builder = Callable[[dict[str, Any]], tuple[dict, str, dict, str]]


def _lr_too_low(hp: dict[str, Any]) -> tuple[dict, str, dict, str]:
    lr = hp["learning_rate"]
    target = min(max(lr * 10, 1e-4), 3e-4)
    if target <= lr:
        raise ValueError(f"learning rate {lr:g} is already at the top of the range")
    return (
        {"learning_rate": target},
        f"learning_rate {lr:g} -> {target:g}",
        {},
        "the failed recipe again with new seeds",
    )


def _too_many_epochs(hp: dict[str, Any]) -> tuple[dict, str, dict, str]:
    n = hp["n_epochs"]
    if n < 2:
        raise ValueError("a single epoch cannot be too many")
    fewer = max(1, round(n / 3))
    control = (
        f"the failed recipe again with new seeds: {n} epochs, more than the target"
    )
    return {"n_epochs": fewer}, f"n_epochs {n} -> {fewer}", {}, control


def _lora_rank_too_small(hp: dict[str, Any]) -> tuple[dict, str, dict, str]:
    r, alpha = hp["lora_r"], hp["lora_alpha"]
    target = min(max(16, r * 4), 64)
    if target <= r:
        raise ValueError(f"rank {r} is already at the top of the range")
    # the same alpha / r as the target, so only the extra rank differs
    same_scale = max(1, round(alpha * r / target))
    control = (
        f"lora_r stays {r}, lora_alpha {alpha:g} -> {same_scale} for the same "
        "alpha / r as the target"
    )
    return (
        {"lora_r": target},
        f"lora_r {r} -> {target}",
        {"lora_alpha": same_scale},
        control,
    )


BUILDERS: dict[str, Builder] = {
    "lr_too_low": _lr_too_low,
    "too_many_epochs": _too_many_epochs,
    "lora_rank_too_small": _lora_rank_too_small,
}


def job_hyperparameters(ctx: InvestigationContext) -> dict[str, Any]:
    return dict(ctx.job_config.get("hyperparameters", ctx.job_config))


def make_pair(
    hypothesis: Hypothesis, ctx: InvestigationContext
) -> tuple[Intervention, Intervention]:
    builder = BUILDERS.get(hypothesis.fault_class)
    if builder is None:
        raise NotImplementedError(f"no intervention for {hypothesis.fault_class}")
    base = job_hyperparameters(ctx)
    t_patch, t_desc, c_patch, c_desc = builder(base)
    target = Intervention(hypothesis.id, "target", t_patch, t_desc)
    control = Intervention(hypothesis.id, "control", c_patch, c_desc)
    t_cost, c_cost = target.relative_cost(base), control.relative_cost(base)
    if c_cost < t_cost and not math.isclose(c_cost, t_cost):
        raise AssertionError("the control must cost at least as much as the target")
    return target, control


def take_fraction(rows: list[Any], fraction: float, seed: int = 0) -> list[Any]:
    """A fixed random share of the rows, the same for every seed of an arm."""
    if fraction >= 1:
        return list(rows)
    keep = max(1, round(len(rows) * fraction))
    idx = sorted(random.Random(seed).sample(range(len(rows)), keep))
    return [rows[i] for i in idx]
