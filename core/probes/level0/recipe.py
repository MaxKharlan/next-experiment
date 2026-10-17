"""Level-0 probes on the job's hyperparameters and checkpoint losses. No GPU, no LLM."""

from __future__ import annotations

import json
from typing import Any

from core.agent.context import InvestigationContext
from core.probes.base import Probe, ProbeResult


def _hp(ctx: InvestigationContext) -> dict[str, Any]:
    return ctx.job_config.get("hyperparameters", {})


def _result(
    probe: Probe, ctx: InvestigationContext, outcome: str, raw: dict
) -> ProbeResult:
    return probe.check_outcome(
        ProbeResult(
            outcome=outcome,
            raw_ref=f"investigations/{ctx.investigation_id}/probes/{probe.name}.json",
            summary=json.dumps(raw),
        )
    )


class LearningRateRange(Probe):
    """Is learning_rate below what LoRA usually needs (Token Factory defaults to 1e-5)?"""

    name = "lr_range"
    level = 0
    outcomes = ("low", "ok")
    LOW = 5e-5

    def estimate_cost(self, ctx: InvestigationContext) -> float:
        return 0.0

    def run(self, ctx: InvestigationContext) -> ProbeResult:
        lr = _hp(ctx)["learning_rate"]
        outcome = "low" if lr < self.LOW else "ok"
        return _result(self, ctx, outcome, {"learning_rate": lr, "low_below": self.LOW})


class EpochCount(Probe):
    name = "epoch_count"
    level = 0
    outcomes = ("high", "ok")
    # calibrated on the dev set: with 200 BANKING77 examples the best recipe trains
    # 6 epochs and 18 did not hurt, so 5 flagged healthy jobs (it was 5 before)
    HIGH = 10

    def estimate_cost(self, ctx: InvestigationContext) -> float:
        return 0.0

    def run(self, ctx: InvestigationContext) -> ProbeResult:
        n = _hp(ctx)["n_epochs"]
        outcome = "high" if n >= self.HIGH else "ok"
        return _result(self, ctx, outcome, {"n_epochs": n, "high_from": self.HIGH})


class LoraRank(Probe):
    name = "lora_rank"
    level = 0
    outcomes = ("low", "ok")
    LOW = 4

    def estimate_cost(self, ctx: InvestigationContext) -> float:
        return 0.0

    def run(self, ctx: InvestigationContext) -> ProbeResult:
        r = _hp(ctx)["lora_r"]
        outcome = "low" if r <= self.LOW else "ok"
        return _result(self, ctx, outcome, {"lora_r": r, "low_up_to": self.LOW})


class LossCurve(Probe):
    """Shape of train_loss and valid_loss over the per-epoch checkpoints.

    Token Factory measures both on a small batch, so the rules use the change from
    the first to the last checkpoint and the gap to the minimum, not single jumps.
    """

    name = "loss_curve"
    level = 0
    outcomes = ("barely_moved", "overfit", "normal")

    def estimate_cost(self, ctx: InvestigationContext) -> float:
        return 0.0

    def run(self, ctx: InvestigationContext) -> ProbeResult:
        points = ctx.job_config.get("checkpoints", [])
        train = [p["train_loss"] for p in points]
        valid = [p["valid_loss"] for p in points]
        if len(train) < 2:
            outcome, drop = "normal", None
        else:
            drop = 1 - train[-1] / train[0]
            if drop < 0.5:
                outcome = "barely_moved"
            elif valid[-1] > 1.2 * min(valid):
                outcome = "overfit"
            else:
                outcome = "normal"
        raw = {"train_loss": train, "valid_loss": valid, "train_drop": drop}
        return _result(self, ctx, outcome, raw)


LEVEL0 = (LearningRateRange(), EpochCount(), LoraRank(), LossCurve())
