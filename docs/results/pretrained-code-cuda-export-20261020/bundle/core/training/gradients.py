"""Inspect gradients after backward, without changing tensors or optimizer state."""


def audit_gradients(model) -> dict:
    import torch

    tensors = []
    counts = dict.fromkeys(("missing", "zero", "nonzero", "nonfinite"), 0)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            status = "missing"
        else:
            values = gradient.detach()
            if values.is_sparse:
                values = values.coalesce().values()
            if not bool(torch.isfinite(values).all()):
                status = "nonfinite"
            elif bool(torch.count_nonzero(values)):
                status = "nonzero"
            else:
                status = "zero"
        counts[status] += 1
        tensors.append({"name": name, "numel": parameter.numel(), "status": status})
    findings = []
    if not tensors:
        findings.append("no_trainable_tensors")
    elif counts["missing"] == len(tensors):
        findings.append("all_gradients_missing")
    elif counts["missing"]:
        findings.append("some_gradients_missing")
    if counts["nonfinite"]:
        findings.append("nonfinite_gradients")
    if tensors and counts["zero"] == len(tensors):
        findings.append("all_gradients_zero")
    return {
        "scope": "first optimizer group after backward, before clipping",
        "tensor_counts": counts,
        "tensors": tensors,
        "findings": findings,
    }
