"""B6: price and duration of a Token Factory fine-tuning job, for the planner.

Prices come from Billing > Prices in the Token Factory console (2026-10-15).
Validated per-model duration fits override the historical Llama-8B fallback.
See docs/b6-post-training.md and docs/results/b467-20261019/cost-fit.json.
Provider capacity waiting is not separately reported by the API; the newer fits
estimate observed stages and cannot guarantee end-to-end completion times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# $ per 1M trained tokens, the same for LoRA and full fine-tuning:
# (context up to 32K, context 65K or 131K). Keys are API model IDs; the IDs of
# Llama-3.1-8B-Instruct and Qwen3-0.6B were checked against the API, the others
# follow the post-training docs and the public model catalog.
FT_PRICES_USD_PER_M: dict[str, tuple[float, float]] = {
    "Qwen/Qwen3-0.6B": (0.40, 1.00),
    "Qwen/Qwen3-1.7B": (0.40, 1.00),
    "Qwen/Qwen3-4B": (0.40, 1.00),
    "Qwen/Qwen3-8B": (0.40, 1.00),
    "Qwen/Qwen3-14B": (0.40, 1.00),
    "meta-llama/Llama-3.2-1B-Instruct": (0.40, 1.00),
    "meta-llama/Llama-3.2-3B-Instruct": (0.40, 1.00),
    "meta-llama/Llama-3.1-8B-Instruct": (0.40, 1.00),
    "unsloth/gpt-oss-20b-BF16": (2.00, 3.00),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (2.80, 5.00),
    "Qwen/Qwen3-32B": (2.80, 5.00),
    "meta-llama/Llama-3.3-70B-Instruct": (2.80, 5.00),
    "Qwen/Qwen3-235B-A22B-Instruct-2507": (5.00, 6.00),
    "unsloth/gpt-oss-120b-BF16": (5.00, 6.00),
}
LONG_CONTEXT = 32_768  # longer contexts cost the second price


@dataclass(frozen=True)
class DurationModel:
    """Stage estimates; queue treatment depends on the fitted source."""

    datasets_s: float = 65  # from submit to both datasets processed
    startup_s: float = 150  # to the "Training started" event
    train_fixed_s: float = 122
    train_s_per_1k_tokens: float = 2.97
    extra_epoch_s: float = 28  # one more checkpoint to save
    finalize_s: float = 75  # from "Training completed" to status succeeded

    def train_seconds(self, tokens: int, epochs: int) -> float:
        return (
            self.train_fixed_s
            + self.train_s_per_1k_tokens * tokens / 1000
            + self.extra_epoch_s * (epochs - 1)
        )

    def total_seconds(self, tokens: int, epochs: int) -> float:
        return (
            self.datasets_s
            + self.startup_s
            + self.train_seconds(tokens, epochs)
            + self.finalize_s
        )


# Fitted on Llama-3.1-8B-Instruct; for other models it is a first guess
LLAMA_8B = DurationModel()

# Validated against the preselected 500-example/two-epoch condition.
# Source: docs/results/b467-20261019/cost-fit.json. Fixed stages include observed
# provider waiting; training-stage validation does not validate total wall time.
MEASURED_DURATIONS: dict[str, DurationModel] = {
    "Qwen/Qwen3-0.6B": DurationModel(
        datasets_s=392.0,
        startup_s=146.5,
        train_fixed_s=97.80460150589496,
        train_s_per_1k_tokens=3.1886777412477776,
        extra_epoch_s=12.200101792117241,
        finalize_s=75.5,
    ),
    "Qwen/Qwen3-1.7B": DurationModel(
        datasets_s=73.0,
        startup_s=151.0,
        train_fixed_s=104.03854946503735,
        train_s_per_1k_tokens=3.1787571255602893,
        extra_epoch_s=12.249258442848724,
        finalize_s=76.5,
    ),
    "meta-llama/Llama-3.2-1B-Instruct": DurationModel(
        datasets_s=43.5,
        startup_s=179.5,
        train_fixed_s=93.43440087572363,
        train_s_per_1k_tokens=2.564999971040886,
        extra_epoch_s=8.017310186091224,
        finalize_s=77.0,
    ),
}


def duration_from_report(report: dict, model: str) -> DurationModel:
    """Accept a model-specific fit only after its preselected validation passes.

    The 30% relative-error cutoff is chosen before collecting this batch's results.
    This estimates observed stage duration, which can include provider queue time.
    Callers explicitly pass the result as estimate_job(..., duration=...).
    """
    if report["model"] != model:
        raise ValueError("Duration fit belongs to another model")
    coefficients = report["coefficients"]
    stages = report["fixed_stages_seconds"]
    values = [*coefficients, *stages.values()]
    if len(coefficients) != 3 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Duration fit contains invalid or negative coefficients")
    checks = report["validation"]
    if not checks or any(
        not math.isfinite(c["absolute_relative_error"])
        or not 0 <= c["absolute_relative_error"] <= 0.30
        for c in checks
    ):
        raise ValueError("Duration fit failed preselected validation")
    return DurationModel(
        **stages,
        train_fixed_s=coefficients[0],
        train_s_per_1k_tokens=coefficients[1],
        extra_epoch_s=coefficients[2],
    )


@dataclass(frozen=True)
class JobEstimate:
    tokens: int
    cost_usd: float
    seconds: float  # default excludes GPU waiting; observed-stage fits may include it


def price_per_m(model: str, context_length: int = 8192) -> float:
    if model not in FT_PRICES_USD_PER_M:
        raise KeyError(f"no fine-tuning price for {model!r}")
    short, long = FT_PRICES_USD_PER_M[model]
    return short if context_length <= LONG_CONTEXT else long


def estimate_job(
    model: str,
    examples: int,
    tokens_per_example: float,
    epochs: int = 1,
    context_length: int = 8192,
    duration: DurationModel | None = None,
) -> JobEstimate:
    """trained_tokens counts every chat template token once per epoch.

    The cost assumes billing uses trained_tokens; check it against Billing > Usage.
    """
    tokens = round(examples * tokens_per_example) * epochs
    duration = duration or MEASURED_DURATIONS.get(model, LLAMA_8B)
    return JobEstimate(
        tokens=tokens,
        cost_usd=tokens * price_per_m(model, context_length) / 1e6,
        seconds=duration.total_seconds(tokens, epochs),
    )
