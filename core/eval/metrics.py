"""Task metrics, keyed by EvaluatorSpec.metric. Each scores one answer from 0 to 1.

Exact match covers intent labels; structured metrics validate the A2/A3 schemas.
"""

from __future__ import annotations

from collections.abc import Callable

from core.eval.structured import function_call_exact, json_fields_exact, json_valid

Metric = Callable[[str, str], float]


def exact_match(answer: str, expected: str) -> float:
    """1 if the answer equals the reference after stripping whitespace at both ends.

    The same comparison as jobs/eval_local/exact_match.py, so scores carry over.
    """
    return float(answer.strip() == expected.strip())


METRICS: dict[str, Metric] = {
    "exact_match": exact_match,
    "json_fields_exact": json_fields_exact,
    "function_call_exact": function_call_exact,
    "json_valid": json_valid,
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(
            f"unknown metric {name!r}; known: {', '.join(sorted(METRICS))}"
        ) from None
