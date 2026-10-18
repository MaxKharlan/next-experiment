"""A13: evaluation of a base model or a Token Factory adapter on a chat JSONL file.

evaluate() returns one record per example: the answer, its score, the prompt and
answer length in tokens and whether generation ended with EOS. breakdown() groups
the errors by input length, reference label and source; compare_checkpoints()
scores every epoch of one job. Importing this package does not import torch.

jobs/eval_local/exact_match.py still serves the existing scripts, the API and the
benchmark; they move here in a separate change, checked to score the same.
"""

from __future__ import annotations

from pathlib import Path

from core.case import EvaluatorSpec
from core.eval.breakdown import breakdown
from core.eval.checkpoints import compare_checkpoints
from core.eval.data import Example, read_examples
from core.eval.generate import DEFAULT_MAX_NEW_TOKENS
from core.eval.metrics import get_metric
from core.eval.records import EvalResult, ExampleResult

__all__ = [
    "EvalResult",
    "Example",
    "ExampleResult",
    "breakdown",
    "compare_checkpoints",
    "evaluate",
    "max_new_tokens",
    "read_examples",
]


def max_new_tokens(spec: EvaluatorSpec) -> int:
    value = spec.parameters.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"max_new_tokens must be a positive integer, got {value!r}")
    return value


def evaluate(
    data: str | Path,
    spec: EvaluatorSpec,
    *,
    adapter: str | Path | None = None,
    base_model: str | None = None,
    template_dir: str | Path | None = None,
    device: str = "auto",
) -> EvalResult:
    """Score an adapter, or a base model alone, on every example of a chat JSONL file.

    The evaluator's metric and max_new_tokens are checked before any model loads.
    """
    from core.eval import generate

    metric = get_metric(spec.metric)
    limit = max_new_tokens(spec)
    examples = read_examples(data)
    device = generate.pick_device(device)
    model, tok, base = generate.load_model(adapter, base_model, template_dir)
    model = model.to(device)
    try:
        results = generate.answers(model, tok, examples, device, metric, limit)
    finally:
        del model
        generate.free_cache(device)
    return EvalResult.from_examples(
        spec.metric,
        results,
        model=base,
        data=str(data),
        max_new_tokens=limit,
        adapter=str(adapter) if adapter is not None else None,
    )
