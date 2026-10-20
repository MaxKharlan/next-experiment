"""Preplanned format and bounded content metrics for development only."""

from __future__ import annotations

import math
import re
from dataclasses import asdict

from core.eval.generate import answer
from core.eval.structured import json_fields_exact, validate_answer


def unwrap(text):
    match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, re.DOTALL)
    return match[1] if match else text


def generation_metrics(text, expected):
    try:
        validate_answer("json_extraction", text)
        valid = 1.0
    except (ValueError, TypeError):
        valid = 0.0
    return {
        "strict_exact": json_fields_exact(text, expected),
        "schema_valid": valid,
        "unwrapped_exact": json_fields_exact(unwrap(text), expected),
    }


def selected_score(scores, correct_index):
    if not scores or not all(math.isfinite(v) for v in scores):
        raise ValueError("Nonfinite or empty choice scores")
    best = max(scores)
    winners = [i for i, value in enumerate(scores) if abs(value - best) <= 1e-6]
    return float(winners == [correct_index])


def completion_logprob(model, prompt_ids, completion_ids, device):
    """Sum log P(c_t | prompt,c_<t); excludes prompt and EOS probabilities."""
    import torch

    if not prompt_ids or not completion_ids:
        raise ValueError("Empty prompt or completion")
    sequence = torch.tensor([prompt_ids + completion_ids], device=device)
    with torch.inference_mode():
        # Only the completion positions are converted to float/log-softmax.
        logits = (
            model(input_ids=sequence, use_cache=False)
            .logits[0, len(prompt_ids) - 1 : -1]
            .float()
        )
        target = torch.tensor(completion_ids, device=device)
        return float(logits.log_softmax(-1).gather(1, target[:, None]).sum().item())


def choice_scores(model, tokenizer, example, choices, device):
    prompt = tokenizer.apply_chat_template(
        example.prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = [tokenizer.encode(c, add_special_tokens=False) for c in choices["choices"]]
    if {len(t) for t in tokens} != {choices["tokens_per_choice"]}:
        raise ValueError("Candidate length changed")
    scores = [completion_logprob(model, prompt_ids, t, device) for t in tokens]
    return {
        "logprobs": scores,
        "choice_exact": selected_score(scores, choices["correct_index"]),
    }


def evaluate_example(model, tokenizer, example, device, choices=None):
    import torch

    with torch.inference_mode():
        result = answer(model, tokenizer, example, device, json_fields_exact, 192)
    row = {**asdict(result), **generation_metrics(result.answer, example.expected)}
    if choices is not None:
        row.update(choice_scores(model, tokenizer, example, choices, device))
    return row


def summary(rows):
    keys = ("strict_exact", "schema_valid", "unwrapped_exact")
    if all("choice_exact" in row for row in rows):
        keys += ("choice_exact",)
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
