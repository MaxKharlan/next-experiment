"""Task metrics, keyed by EvaluatorSpec.metric. Each scores one answer from 0 to 1.

Only exact match exists so far: it is the BANKING77 metric (A1). The JSON
extraction (A2) and function calling (A3) metrics join when their datasets do.
"""

from __future__ import annotations

from collections.abc import Callable

Metric = Callable[[str, str], float]


def exact_match(answer: str, expected: str) -> float:
    """1 if the answer equals the reference after stripping whitespace at both ends.

    The same comparison as jobs/eval_local/exact_match.py, so scores carry over.
    """
    return float(answer.strip() == expected.strip())


METRICS: dict[str, Metric] = {"exact_match": exact_match}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(
            f"unknown metric {name!r}; known: {', '.join(sorted(METRICS))}"
        ) from None
