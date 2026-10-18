"""Mode B fault 10: token-value padding masks remove real EOS targets."""

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


def audit_eos_labels(reference: list[dict], observed: list[dict], eos_id: int) -> dict:
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
        for expected, actual in zip(ref["labels"][1:], obs["labels"][1:]):
            if actual not in (expected, -100):
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
