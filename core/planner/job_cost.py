"""B6: price and duration of a Token Factory fine-tuning job, for the planner.

Prices come from Billing > Prices in the Token Factory console (2026-10-15). The
duration model is fitted on four Llama-3.1-8B-Instruct LoRA jobs, see
docs/b6-post-training.md. It leaves out the wait for GPU capacity: the API does not
show it, a waiting job looks like one that has just started, and on 2026-10-15 the
wait took from zero to over two hours.
"""

from __future__ import annotations

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
    """Seconds per stage of a job that got a GPU right away."""

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


@dataclass(frozen=True)
class JobEstimate:
    tokens: int
    cost_usd: float
    seconds: float  # without the wait for GPU capacity


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
    duration: DurationModel = LLAMA_8B,
) -> JobEstimate:
    """trained_tokens counts every chat template token once per epoch.

    The cost assumes billing uses trained_tokens; check it against Billing > Usage.
    """
    tokens = round(examples * tokens_per_example) * epochs
    return JobEstimate(
        tokens=tokens,
        cost_usd=tokens * price_per_m(model, context_length) / 1e6,
        seconds=duration.total_seconds(tokens, epochs),
    )
