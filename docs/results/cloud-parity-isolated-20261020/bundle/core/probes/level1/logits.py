"""Teacher-forced token diagnostics; no fitted diagnosis or causal verdict."""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.eval.data import Example


@dataclass(frozen=True)
class TokenRegions:
    input_ids: tuple[int, ...]
    prompt: tuple[int, ...]
    answer: tuple[int, ...]
    template_special: tuple[int, ...]
    answer_end: int
    eos_ids: tuple[int, ...]


def encode_regions(tokenizer, example: Example, *, context_length: int) -> TokenRegions:
    if type(context_length) is not int or context_length < 2:
        raise ValueError("An explicit positive context limit is required")
    prompt_text = tokenizer.apply_chat_template(
        example.prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        [*example.prompt, {"role": "assistant", "content": example.expected}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prefix = tokenizer.encode(prompt_text, add_special_tokens=False)
    full = tokenizer.encode(full_text, add_special_tokens=False)
    if not prefix or full[: len(prefix)] != prefix:
        raise ValueError("Prompt is not a token prefix of the full rendered chat")
    if len(full) > context_length:
        raise ValueError("Context overflow; truncation is forbidden")
    eos = tokenizer.eos_token_id
    eos_ids = (eos,) if type(eos) is int else tuple(eos or ())
    if not eos_ids:
        raise ValueError("Tokenizer must declare EOS")
    ending = next(
        (i for i in range(len(prefix), len(full)) if full[i] in eos_ids), None
    )
    if ending is None or ending == len(prefix):
        raise ValueError("Rendered answer needs body tokens followed by EOS")
    specials = set(tokenizer.all_special_ids)
    template = tuple(i for i in range(1, len(full)) if full[i] in specials)
    prompt = tuple(i for i in range(1, len(prefix)) if i not in template)
    answer = tuple(i for i in range(len(prefix), ending) if i not in template)
    if not answer:
        raise ValueError("No answer-body tokens")
    return TokenRegions(tuple(full), prompt, answer, template, ending, eos_ids)


def validate_regions(regions: TokenRegions, vocabulary: int):
    ids = regions.input_ids
    if len(ids) < 2 or any(type(i) is not int or not 0 <= i < vocabulary for i in ids):
        raise ValueError("Invalid input token IDs")
    positions = (*regions.prompt, *regions.answer, *regions.template_special)
    if len(positions) != len(set(positions)) or any(
        type(i) is not int or not 1 <= i < len(ids) for i in positions
    ):
        raise ValueError("Token regions overlap or include unpredicted positions")
    if not regions.answer or not 1 <= regions.answer_end < len(ids):
        raise ValueError("Missing answer or invalid EOS boundary")
    if max(regions.answer) >= regions.answer_end:
        raise ValueError("Answer-body tokens must precede EOS")
    if (
        not regions.eos_ids
        or len(set(regions.eos_ids)) != len(regions.eos_ids)
        or any(type(i) is not int or not 0 <= i < vocabulary for i in regions.eos_ids)
    ):
        raise ValueError("Invalid EOS IDs")
    if ids[regions.answer_end] not in regions.eos_ids:
        raise ValueError("EOS boundary does not point to an EOS token")


def loss_summary(total_nll: float, tokens: int):
    mean = total_nll / tokens if tokens else None
    overflow = mean is not None and mean > 700
    return {
        "tokens": tokens,
        "total_nll": total_nll,
        "mean_nll": mean,
        "perplexity": math.exp(mean) if mean is not None and not overflow else None,
        "perplexity_overflow": overflow,
    }


def inspect_logits(logits, regions: TokenRegions):
    """Position i is predicted by logits[i-1], including the EOS boundary."""
    import torch

    if logits.ndim != 2 or logits.shape[0] != len(regions.input_ids):
        raise ValueError("Expected sequence × vocabulary logits")
    validate_regions(regions, logits.shape[1])
    positions = sorted(
        {
            *regions.prompt,
            *regions.answer,
            *regions.template_special,
            regions.answer_end,
        }
    )
    selected = logits[[i - 1 for i in positions]].float()
    if not torch.isfinite(selected).all():
        raise ValueError("Nonfinite logits")
    logprobs = selected.log_softmax(-1)
    lookup = {position: j for j, position in enumerate(positions)}
    out = {}
    for name in ("prompt", "answer", "template_special"):
        group = getattr(regions, name)
        total = sum(
            -float(logprobs[lookup[i], regions.input_ids[i]].item()) for i in group
        )
        out[name] = loss_summary(total, len(group))
    eos_probability = float(
        logprobs[lookup[regions.answer_end], list(regions.eos_ids)].exp().sum().item()
    )
    out.update(
        eos_probability_after_reference=eos_probability,
        total_input_tokens=len(regions.input_ids),
    )
    return out


def inspect_example(model, tokenizer, example, device, *, context_length):
    import torch

    regions = encode_regions(tokenizer, example, context_length=context_length)
    ids = torch.tensor([regions.input_ids], device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits[0]
        result = inspect_logits(logits, regions)
    return {"line": example.line, "source": example.source, **result}


def aggregate(rows):
    if not rows:
        raise ValueError("No logit measurements")
    result = {
        name: loss_summary(
            sum(row[name]["total_nll"] for row in rows),
            sum(row[name]["tokens"] for row in rows),
        )
        for name in ("prompt", "answer", "template_special")
    }
    result.update(
        examples=len(rows),
        mean_eos_probability_after_reference=sum(
            row["eos_probability_after_reference"] for row in rows
        )
        / len(rows),
        scope="Teacher-forced references; special-template tokens only. No generated-EOS frequency or calibrated fault likelihood.",
    )
    return result
