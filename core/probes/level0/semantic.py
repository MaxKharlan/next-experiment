"""Embedding review candidates. Thresholds are prospective, not fitted likelihoods."""

from __future__ import annotations

import numpy as np


def unit_vectors(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all() or min(array.shape) < 1:
        raise ValueError("Expected a finite, nonempty embedding matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError("Zero embedding")
    return array / norms


def neighbor_audit(
    reference,
    audit,
    reference_vectors,
    audit_vectors,
    allowed_labels,
    *,
    k=3,
    similarity_min=0.8,
):
    if (
        type(k) is not int
        or k < 1
        or k > len(reference)
        or not 0 <= similarity_min <= 1
    ):
        raise ValueError("Invalid neighbor parameters")
    left, right = unit_vectors(reference_vectors), unit_vectors(audit_vectors)
    if (
        len(left) != len(reference)
        or len(right) != len(audit)
        or left.shape[1] != right.shape[1]
    ):
        raise ValueError("Rows or dimensions differ")
    if any(row["label"] not in allowed_labels for row in reference):
        raise ValueError("Invalid reference label")
    similarities = right @ left.T
    output = []
    for row, scores in zip(audit, similarities, strict=True):
        indices = np.argsort(-scores, kind="stable")[:k]
        neighbors = [
            {
                "id": reference[i]["id"],
                "text": reference[i]["text"],
                "label": reference[i]["label"],
                "similarity": float(scores[i]),
            }
            for i in indices
        ]
        bad_format = row["observed_label"] not in allowed_labels
        disagree = sum(
            n["similarity"] >= similarity_min and n["label"] != row["observed_label"]
            for n in neighbors
        )
        output.append(
            {
                "id": row["id"],
                "format_error": bad_format,
                "contradiction": not bad_format and disagree >= 2,
                "neighbors": neighbors,
            }
        )
    return output


def confusion(truth, predicted):
    if (
        len(truth) != len(predicted)
        or not truth
        or any(type(x) is not bool for x in (*truth, *predicted))
    ):
        raise ValueError("Expected equally sized nonempty boolean labels")
    tp = sum(t and p for t, p in zip(truth, predicted, strict=True))
    fp = sum(not t and p for t, p in zip(truth, predicted, strict=True))
    fn = sum(t and not p for t, p in zip(truth, predicted, strict=True))
    tn = len(truth) - tp - fp - fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "accuracy": (tp + tn) / len(truth),
        "examples": len(truth),
    }
