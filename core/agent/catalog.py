"""The fault catalog from section 08 of the project doc, as the agent sees it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fault:
    number: int
    fault_class: str  # a value of core.agent.schemas.FaultClass
    group: str  # data, recipe or code
    cause: str
    symptom: str
    modes: str  # "AB": data and recipe faults; "B": training-code faults


FAULTS: tuple[Fault, ...] = (
    Fault(
        1,
        "truncation",
        "data",
        "context_length is shorter than some examples, so they are cut",
        "answers stop early, JSON breaks on long inputs",
        "AB",
    ),
    Fault(
        2,
        "chat_template_mismatch",
        "data",
        "the chat template in training differs from the one at inference",
        "the model ignores the instruction or adds extra tokens",
        "AB",
    ),
    Fault(
        3,
        "contradictory_labels",
        "data",
        "similar inputs have conflicting answers in the data",
        "unstable answers, valid_loss plateaus",
        "AB",
    ),
    Fault(
        4,
        "train_val_overlap",
        "data",
        "the validation file overlaps the training file",
        "low valid_loss but poor quality on held-out data",
        "AB",
    ),
    Fault(
        5,
        "distribution_shift",
        "data",
        "training examples do not look like production requests",
        "good on validation, poor on real requests",
        "AB",
    ),
    Fault(
        6,
        "lr_too_low",
        "recipe",
        "the learning rate is too small for LoRA",
        "the model barely differs from the base model",
        "AB",
    ),
    Fault(
        7,
        "too_many_epochs",
        "recipe",
        "too many epochs",
        "valid_loss rises after its minimum, the model repeats training answers",
        "AB",
    ),
    Fault(
        8,
        "lora_rank_too_small",
        "recipe",
        "the LoRA rank is too small for the task",
        "quality stalls below the expected level",
        "AB",
    ),
    Fault(
        9,
        "loss_on_prompt",
        "code",
        "the loss includes prompt tokens",
        "the model repeats parts of the prompt",
        "B",
    ),
    Fault(
        10,
        "pad_eos_masking",
        "code",
        "pad_token equals eos_token and pad is masked in the labels",
        "the model does not stop generating",
        "B",
    ),
    Fault(
        11,
        "double_label_shift",
        "code",
        "labels are shifted twice",
        "loss plateaus above normal, answers are off by one token",
        "B",
    ),
    Fault(
        12,
        "grad_accum_normalization",
        "code",
        "with gradient accumulation the loss is averaged per micro-batch",
        "a worse loss curve than the same config without accumulation",
        "B",
    ),
)


def allowed_classes(mode: str) -> set[str]:
    return {f.fault_class for f in FAULTS if mode in f.modes}


def catalog_text(mode: str) -> str:
    return "\n".join(
        f"- {f.fault_class} ({f.group}): {f.cause}. Symptom: {f.symptom}."
        for f in FAULTS
        if mode in f.modes
    )
