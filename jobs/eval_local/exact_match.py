"""Exact match of a LoRA adapter on a chat JSONL file, on a MacBook (MPS) or CPU.

A stand-in for core/eval (A13). Loads the base model named in adapter_config.json,
applies the LoRA weights from a directory (a Token Factory checkpoint or a
train_lora.py run) and compares greedy answers with the reference. If the
checkpoint has a chat_template.jinja, that template is used, so prompts match
training.

Token Factory saves weight names without PEFT's "base_model.model." prefix, so
PeftModel.from_pretrained would skip them all. Its configs also set
init_lora_weights to false, which leaves random weights in their place. The weights
are therefore loaded here by name, and a missing LoRA weight is an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from jobs.kaggle.train_lora import exact_match, pick_device, read_jsonl

PREFIX = "base_model.model."


def peft_names(weights: dict[str, Any]) -> dict[str, Any]:
    """Weight names as PEFT saves them, whether or not they have the prefix."""
    return {k if k.startswith(PREFIX) else PREFIX + k: v for k, v in weights.items()}


def load_adapter(adapter_dir: Path, base_model: str) -> Any:
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


def evaluate_adapter(
    adapter_dir: str | Path,
    valid_path: str | Path,
    base_model: str | None = None,
    device: str = "auto",
) -> float:
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    base_model = base_model or config["base_model_name_or_path"]
    device = pick_device(device)
    tok = AutoTokenizer.from_pretrained(base_model)
    template = adapter_dir / "chat_template.jinja"
    if template.exists():
        tok.chat_template = template.read_text(encoding="utf-8")
    model = load_adapter(adapter_dir, base_model).to(device)
    return exact_match(model, tok, read_jsonl(valid_path), device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dir")
    parser.add_argument("--valid", required=True)
    parser.add_argument("--base-model")
    args = parser.parse_args()
    score = evaluate_adapter(args.adapter_dir, args.valid, args.base_model)
    print(json.dumps({"adapter": args.adapter_dir, "exact_match": score}))


if __name__ == "__main__":
    main()
