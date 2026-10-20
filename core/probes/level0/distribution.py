"""Declared-target label divergence and held-out embedding two-sample checks."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from core.probes.level0.semantic import unit_vectors


def label_divergence(labels, target):
    if (
        not labels
        or not target
        or any(not math.isfinite(p) or p < 0 for p in target.values())
        or not math.isclose(sum(target.values()), 1, abs_tol=1e-8)
    ):
        raise ValueError("Require observed labels and normalized target probabilities")
    counts = Counter(labels)
    if set(counts) - set(target):
        raise ValueError("Observed label absent from declared target vocabulary")
    observed = {label: counts[label] / len(labels) for label in target}
    js = 0.0
    for label, q in target.items():
        p = observed[label]
        mean = (p + q) / 2
        if p:
            js += 0.5 * p * math.log2(p / mean)
        if q:
            js += 0.5 * q * math.log2(q / mean)
    return {
        "js_divergence_bits": js,
        "observed": observed,
        "target": dict(target),
        "examples": len(labels),
    }


def two_sample_auc(left, right, *, left_ids, right_ids, seed=5101):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    a, b = unit_vectors(left), unit_vectors(right)
    if a.shape[1] != b.shape[1] or len(a) < 5 or len(b) < 5:
        raise ValueError("Need equal dimensions and at least five rows per source")
    if (
        len(a) != len(left_ids)
        or len(b) != len(right_ids)
        or len(set(left_ids)) != len(a)
        or len(set(right_ids)) != len(b)
        or set(left_ids) & set(right_ids)
    ):
        raise ValueError("Require distinct nonoverlapping query IDs")
    x = np.concatenate([a, b])
    y = np.array([0] * len(a) + [1] * len(b))
    probabilities = cross_val_predict(
        LogisticRegression(C=1, max_iter=1000),
        x,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    return {
        "auc": float(roc_auc_score(y, probabilities)),
        "left": len(a),
        "right": len(b),
        "folds": 5,
        "seed": seed,
        "probabilities": probabilities.tolist(),
        "scope": "Separability of these samples; neither a cause nor calibrated fault likelihood.",
    }
