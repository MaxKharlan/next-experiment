"""Inventory trainable tensors in the local LoRA training configuration."""


def audit_trainable_parameters(model) -> dict:
    """Count unique tensors using named_parameters' standard deduplication.

    LoRA membership uses exact path components used by PEFT's A/B modules.
    Other trainable parameters are reported for review, not silently frozen.
    """
    total = trainable = lora_total = lora_trainable = 0
    tensors = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        is_lora = bool({"lora_A", "lora_B"} & set(name.split(".")))
        total += count
        if is_lora:
            lora_total += count
        if parameter.requires_grad:
            trainable += count
            if is_lora:
                lora_trainable += count
            tensors.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "numel": count,
                    "kind": "lora" if is_lora else "other",
                }
            )
    findings = []
    if trainable == 0:
        findings.append("no_trainable_parameters")
    if lora_trainable == 0:
        findings.append("no_trainable_lora_parameters")
    if trainable > lora_trainable:
        findings.append("trainable_non_lora_parameters")
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_fraction": trainable / total if total else None,
        "lora_parameters": lora_total,
        "trainable_lora_parameters": lora_trainable,
        "trainable_non_lora_parameters": trainable - lora_trainable,
        "trainable_tensors": tensors,
        "findings": findings,
        "scope": "PEFT linear LoRA A/B with bias=none; snapshot before training",
    }
