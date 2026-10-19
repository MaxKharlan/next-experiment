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
from core.eval.structured import (
    TASK_EVALUATORS,
    structured_summary,
    validate_answer,
    validate_evaluator,
)

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
    runtime: dict | None = None,
) -> EvalResult:
    """Score an adapter, or a base model alone, on every example of a chat JSONL file.

    The evaluator's metric and max_new_tokens are checked before any model loads.
    """
    from core.eval import generate

    task = next(
        (
            task
            for task, (_, metric, _) in TASK_EVALUATORS.items()
            if metric == spec.metric and task != "banking77"
        ),
        None,
    )
    if task:
        spec = validate_evaluator(task, spec)
    metric = get_metric(spec.metric)
    limit = max_new_tokens(spec)
    from core.eval.runtime import file_digest

    data_hash = file_digest(data)
    examples = read_examples(data)
    if task:
        for example in examples:
            validate_answer(task, example.expected)
    frozen = None
    adapter_hashes = {}
    template_source = adapter if adapter is not None else template_dir
    if runtime is not None:
        from core.eval.runtime import verify_runtime

        frozen = verify_runtime(runtime, model=base_model, template_dir=template_source)
        base_model = frozen.snapshot
        if adapter is not None:
            adapter_hashes = {
                name: file_digest(Path(adapter) / name)
                for name in ("adapter_model.safetensors", "adapter_config.json")
            }
    device = generate.pick_device(device)
    model, tok, base = generate.load_model(adapter, base_model, template_dir)
    model = model.to(device)
    try:
        results = generate.answers(model, tok, examples, device, metric, limit)
    finally:
        del model
        generate.free_cache(device)
    if file_digest(data) != data_hash:
        raise ValueError("Evaluation dataset changed during scoring")
    for name, expected in adapter_hashes.items():
        if file_digest(Path(adapter) / name) != expected:
            raise ValueError("Adapter changed during scoring")
    if frozen is not None:
        verify_runtime(frozen, template_dir=template_source)
        base = frozen.model
    result = EvalResult.from_examples(
        spec.metric,
        results,
        model=base,
        data=str(data),
        max_new_tokens=limit,
        adapter=str(adapter) if adapter is not None else None,
        runtime=frozen.model_dump() if frozen is not None else None,
        data_sha256=data_hash,
        adapter_files_sha256=adapter_hashes,
    )

    if task:
        result.details = structured_summary(
            task, [r.answer for r in results], [r.expected for r in results]
        )
    return result
