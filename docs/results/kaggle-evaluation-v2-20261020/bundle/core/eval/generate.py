"""Greedy answers of a base model or a LoRA adapter, one record per example.

Runs on a MacBook (MPS), CUDA or CPU. torch, transformers and peft are imported
only by the loaders, so the generation loop runs in tests with a stand-in model.

Prompts and decoding match jobs/eval_local/exact_match.py, so the scores of the
published benchmark carry over: the chat template of the checkpoint when it has a
chat_template.jinja, thinking off, the generation prompt added, greedy decoding
and at most 32 new tokens unless the evaluator sets max_new_tokens.

Token Factory saves weight names without PEFT's "base_model.model." prefix and sets
init_lora_weights to false, so PeftModel.from_pretrained would leave random weights
in place. load_adapter loads the weights by name, and a missing LoRA weight is an
error.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.eval.data import Example
from core.eval.metrics import Metric
from core.eval.records import ExampleResult

DEFAULT_MAX_NEW_TOKENS = 32
PREFIX = "base_model.model."


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def free_cache(device: str) -> None:
    if device == "mps":  # without this the MPS cache fills unified memory
        import torch

        torch.mps.empty_cache()


def peft_names(weights: dict[str, Any]) -> dict[str, Any]:
    """Weight names as PEFT saves them, whether or not they have the prefix."""
    return {k if k.startswith(PREFIX) else PREFIX + k: v for k, v in weights.items()}


def load_tokenizer(base_model: str, template_dir: Path | None = None) -> Any:
    """The base model's tokenizer, with a checkpoint's chat template if it has one."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    template = template_dir / "chat_template.jinja" if template_dir else None
    if template is not None and template.exists():
        tok.chat_template = template.read_text(encoding="utf-8")
    return tok


def load_adapter(adapter_dir: Path, base_model: str) -> Any:
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    config = LoraConfig.from_pretrained(adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    model = get_peft_model(model, config)
    weights = peft_names(load_file(adapter_dir / "adapter_model.safetensors"))
    loaded = set_peft_model_state_dict(model, weights)
    missing = [k for k in loaded.missing_keys if "lora_" in k]
    if missing or loaded.unexpected_keys:
        raise ValueError(
            f"{adapter_dir}: {len(missing)} LoRA weights missing, "
            f"{len(loaded.unexpected_keys)} unexpected, e.g. "
            f"{(missing + list(loaded.unexpected_keys))[:2]}"
        )
    return model


def load_base(base_model: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)


def load_model(
    adapter: str | Path | None = None,
    base_model: str | None = None,
    template_dir: str | Path | None = None,
) -> tuple[Any, Any, str]:
    """Model, tokenizer and base model name for an adapter or for a base model alone.

    An adapter names its base model in adapter_config.json; base_model overrides it.
    A base model alone uses template_dir's chat template, so it is prompted like the
    adapters it is compared with.
    """
    if adapter is not None:
        adapter = Path(adapter)
        config = json.loads((adapter / "adapter_config.json").read_text())
        base = base_model or config["base_model_name_or_path"]
        return load_adapter(adapter, base), load_tokenizer(base, adapter), base
    if not base_model:
        raise ValueError("give an adapter directory or a base model")
    template = Path(template_dir) if template_dir else None
    return load_base(base_model), load_tokenizer(base_model, template), base_model


def eos_ids(model: Any, tok: Any) -> set[int]:
    """Token ids that end generation: the model's generation config and the tokenizer's."""
    ids: set[int] = set()
    config = getattr(model, "generation_config", None)
    for value in (getattr(config, "eos_token_id", None), tok.eos_token_id):
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, Iterable):
            ids.update(int(v) for v in value)
    return ids


def _ids(sequence: Any) -> list[int]:
    return [
        int(t) for t in (sequence.tolist() if hasattr(sequence, "tolist") else sequence)
    ]


def answer(
    model: Any,
    tok: Any,
    example: Example,
    device: str,
    metric: Metric,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    stop_ids: set[int] | None = None,
) -> ExampleResult:
    """The model's greedy answer to one example, scored, with how generation ended."""
    prompt = tok.apply_chat_template(
        example.prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_tokens = len(_ids(ids["input_ids"][0]))
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0][prompt_tokens:]
    text = tok.decode(new, skip_special_tokens=True)
    new_ids = _ids(new)
    stop_ids = eos_ids(model, tok) if stop_ids is None else stop_ids
    stopped = bool(new_ids) and new_ids[-1] in stop_ids
    return ExampleResult(
        line=example.line,
        expected=example.expected,
        answer=text,
        score=metric(text, example.expected),
        prompt_tokens=prompt_tokens,
        answer_tokens=len(new_ids) - stopped,
        stopped_on_eos=stopped,
        hit_limit=not stopped and len(new_ids) >= max_new_tokens,
        source=example.source,
    )


def answers(
    model: Any,
    tok: Any,
    examples: list[Example],
    device: str,
    metric: Metric,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> list[ExampleResult]:
    model.eval()
    stop_ids = eos_ids(model, tok)
    results = [
        answer(model, tok, e, device, metric, max_new_tokens, stop_ids)
        for e in examples
    ]
    free_cache(device)
    return results
