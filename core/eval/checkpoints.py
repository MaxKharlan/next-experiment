"""Quality of every checkpoint of one job on the same examples.

Token Factory saves one checkpoint per epoch, so the n-th checkpoint by step is
epoch n. A score that peaks and then falls points at fault 7 (too many epochs);
Token Factory's own valid_loss is measured on a small batch and too noisy for that.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from core.eval.breakdown import stopping
from core.eval.records import EvalResult


def compare_checkpoints(
    checkpoints: Iterable[tuple[int, str | Path]],
    score: Callable[[Path], EvalResult],
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Score each (step, adapter directory) and name the best one.

    score evaluates one adapter directory, e.g.
    lambda d: evaluate(data, spec, adapter=d), so every checkpoint sees the same
    examples and the same decoding.
    """
    ordered = sorted(checkpoints, key=lambda item: item[0])
    if not ordered:
        raise ValueError("no checkpoints to compare")
    rows = []
    for epoch, (step, path) in enumerate(ordered, start=1):
        result = score(Path(path))
        ends = stopping(result.examples)
        rows.append(
            {
                "step": step,
                "epoch": epoch,
                "adapter": str(path),
                "value": result.value,
                "no_eos_share": ends["no_eos_share"],
                "hit_limit_share": ends["hit_limit_share"],
            }
        )
    pick = max if higher_is_better else min
    best = pick(rows, key=lambda r: r["value"])
    return {
        "metric": result.metric,
        "rows": rows,
        "best_step": best["step"],
        "best_epoch": best["epoch"],
        "last_is_best": best is rows[-1],
    }
