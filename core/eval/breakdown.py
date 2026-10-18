"""Where a model's errors are: by input length, by reference label, by source.

Also how its answers end: with EOS, or cut at max_new_tokens. A model that never
stops (fault 10 in mode B) or stops too early (fault 1) shows here before any metric
explains why. No torch, no LLM: only the records of an evaluation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from core.eval.records import EvalResult, ExampleResult

LENGTH_BUCKETS = 4  # quartiles of prompt length in tokens


def _group(
    examples: Iterable[ExampleResult], key: Callable[[ExampleResult], Any]
) -> list[dict[str, Any]]:
    groups: dict[Any, list[ExampleResult]] = defaultdict(list)
    for example in examples:
        groups[key(example)].append(example)
    rows = []
    for value, members in groups.items():
        errors = sum(not e.correct for e in members)
        rows.append(
            {
                "value": value,
                "examples": len(members),
                "errors": errors,
                "error_rate": errors / len(members),
            }
        )
    return sorted(rows, key=lambda r: (-r["errors"], -r["examples"], str(r["value"])))


def by_prompt_length(
    examples: list[ExampleResult], buckets: int = LENGTH_BUCKETS
) -> list[dict[str, Any]]:
    """Error rate per bucket of prompt tokens; edges are nearest-rank quantiles."""
    lengths = sorted(e.prompt_tokens for e in examples)
    edges = sorted(
        {
            lengths[max(0, math.ceil(i / buckets * len(lengths)) - 1)]
            for i in range(1, buckets + 1)
        }
    )
    rows = []
    low = lengths[0]
    for high in edges:
        members = [e for e in examples if low <= e.prompt_tokens <= high]
        errors = sum(not e.correct for e in members)
        rows.append(
            {
                "tokens": [low, high],
                "examples": len(members),
                "errors": errors,
                "error_rate": errors / len(members) if members else 0.0,
            }
        )
        low = high + 1
    return rows


def by_label(examples: list[ExampleResult]) -> list[dict[str, Any]]:
    """Errors per reference answer: the type of input for a classification task."""
    return _group(examples, lambda e: e.expected.strip())


def by_source(examples: list[ExampleResult]) -> list[dict[str, Any]]:
    """Errors per source, for records that name one; empty when none does."""
    if all(e.source is None for e in examples):
        return []
    return _group(examples, lambda e: e.source)


def stopping(examples: list[ExampleResult]) -> dict[str, Any]:
    """How answers end: with EOS, cut at the limit, and how long they are."""
    n = len(examples)
    tokens = sorted(e.answer_tokens for e in examples)
    return {
        "examples": n,
        "no_eos_share": sum(not e.stopped_on_eos for e in examples) / n,
        "hit_limit_share": sum(e.hit_limit for e in examples) / n,
        "answer_tokens_mean": sum(tokens) / n,
        "answer_tokens_max": tokens[-1],
    }


def breakdown(result: EvalResult) -> dict[str, Any]:
    examples = result.examples
    if not examples:
        raise ValueError("no examples to break down")
    return {
        "metric": result.metric,
        "value": result.value,
        "errors": len(result.wrong()),
        "by_prompt_length": by_prompt_length(examples),
        "by_label": by_label(examples),
        "by_source": by_source(examples),
        "stopping": stopping(examples),
    }
