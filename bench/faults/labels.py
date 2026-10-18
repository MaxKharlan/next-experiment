"""Mode B label faults and audits against healthy answer-only labels."""

from copy import deepcopy


def mask_padding_labels(rows: list[dict], pad_id: int) -> list[dict]:
    """Inject token-value masking on a copy, including non-padding positions."""
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError("pad_id must be a nonnegative integer")
    result = deepcopy(rows)
    for row in result:
        if len(row["input_ids"]) != len(row["labels"]):
            raise ValueError("input_ids and labels must have equal lengths")
        row["labels"] = [
            -100 if token == pad_id else label
            for token, label in zip(row["input_ids"], row["labels"])
        ]
    return result


def audit_eos_labels(
    reference: list[dict],
    observed: list[dict],
    eos_id: int,
    *,
    allow_prompt_targets: bool = False,
) -> dict:
    """Count removed EOS targets against healthy labels after causal shifting.

    Prompt tokens and the first position are not supervised targets. Absence of
    EOS in the reference is inconclusive, including after context truncation.
    """
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("eos_id must be a nonnegative integer")
    if len(reference) != len(observed):
        raise ValueError("Reference and observed rows must align")
    eos_targets = removed = before = after = 0
    for ref, obs in zip(reference, observed):
        if (
            ref["input_ids"] != obs["input_ids"]
            or len(ref["labels"]) != len(ref["input_ids"])
            or len(obs["labels"]) != len(ref["labels"])
        ):
            raise ValueError("Reference and observed tokens must align")
        for token, expected, actual in zip(
            ref["input_ids"][1:], ref["labels"][1:], obs["labels"][1:]
        ):
            added_prompt = allow_prompt_targets and expected == -100 and actual == token
            if actual not in (expected, -100) and not added_prompt:
                raise ValueError("Observed labels must only mask reference labels")
            before += expected != -100
            after += actual != -100
            if expected == eos_id:
                eos_targets += 1
                removed += actual == -100
    return {
        "eos_token_id": eos_id,
        "reference_targets": before,
        "observed_targets": after,
        "reference_eos_targets": eos_targets,
        "removed_eos_targets": removed,
        "removed_eos_fraction": removed / eos_targets if eos_targets else None,
        "status": "no_eos_targets"
        if not eos_targets
        else "removed"
        if removed
        else "preserved",
    }


def include_prompt_labels(rows: list[dict]) -> list[dict]:
    """Inject fault 9 on unpadded encoded rows without mutating the reference."""
    result = deepcopy(rows)
    for row in result:
        if len(row["input_ids"]) != len(row["labels"]):
            raise ValueError("input_ids and labels must have equal lengths")
        row["labels"] = list(row["input_ids"])
    return result


def audit_prompt_labels(reference: list[dict], observed: list[dict]) -> dict:
    """Audit unpadded answer-only rows; ignored reference positions are prompt.

    Position zero is excluded because causal loss has no prediction for it.
    Observed labels may only reveal original token IDs at ignored positions.
    """
    if len(reference) != len(observed):
        raise ValueError("Reference and observed rows must align")
    prompt = included = answer = 0
    for ref, obs in zip(reference, observed):
        if (
            ref["input_ids"] != obs["input_ids"]
            or len(ref["labels"]) != len(ref["input_ids"])
            or len(obs["labels"]) != len(ref["labels"])
        ):
            raise ValueError("Reference and observed tokens must align")
        for token, expected, actual in zip(
            ref["input_ids"][1:], ref["labels"][1:], obs["labels"][1:]
        ):
            if expected == -100:
                if actual not in (-100, token):
                    raise ValueError(
                        "Prompt labels must match input tokens or be ignored"
                    )
                prompt += 1
                included += actual != -100
            else:
                if actual != expected:
                    raise ValueError("Answer labels must remain unchanged")
                answer += 1
    total = answer + included
    return {
        "prompt_positions": prompt,
        "supervised_prompt_tokens": included,
        "answer_targets": answer,
        "total_targets": total,
        "prompt_loss_fraction": included / total if total else None,
        "status": "included"
        if included
        else "masked"
        if prompt
        else "no_prompt_targets",
    }


def shift_labels_once(rows: list[dict]) -> list[dict]:
    """Pre-shift unpadded labels, causing a second shift inside causal loss."""
    result = deepcopy(rows)
    for row in result:
        if len(row["input_ids"]) != len(row["labels"]):
            raise ValueError("input_ids and labels must have equal lengths")
        labels = row["labels"]
        row["labels"] = labels[1:] + [-100] if labels else []
    return result


def audit_label_shift(reference: list[dict], observed: list[dict]) -> dict:
    """Compare effective targets with healthy and pre-shifted reference labels.

    A matching pre-shift pattern is evidence about these tensors, not a diagnosis
    of downstream model behavior. Identical candidate patterns are inconclusive.
    """
    if len(reference) != len(observed):
        raise ValueError("Reference and observed rows must align")
    healthy_errors = shifted_errors = informative = targets = 0
    for ref, obs in zip(reference, observed):
        labels = ref["labels"]
        if (
            ref["input_ids"] != obs["input_ids"]
            or len(labels) != len(ref["input_ids"])
            or len(obs["labels"]) != len(labels)
        ):
            raise ValueError("Reference and observed tokens must align")
        shifted = labels[1:] + [-100] if labels else []
        for healthy, faulty, actual in zip(labels[1:], shifted[1:], obs["labels"][1:]):
            healthy_errors += actual != healthy
            shifted_errors += actual != faulty
            informative += healthy != faulty
            targets += actual != -100
    status = (
        "inconclusive"
        if not informative
        else "aligned"
        if not healthy_errors
        else "double_shift"
        if not shifted_errors
        else "other_mismatch"
    )
    return {
        "status": status,
        "informative_positions": informative,
        "healthy_mismatches": healthy_errors,
        "preshift_mismatches": shifted_errors,
        "observed_targets": targets,
    }
