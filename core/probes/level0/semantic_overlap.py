"""Conservative context guards around symmetric embedding overlap candidates."""

from __future__ import annotations

import numpy as np

from core.probes.level0.overlap import normalize
from core.probes.level0.semantic import unit_vectors


def compare_prompts(left, right, left_vector, right_vector, *, threshold=0.92):
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid cosine threshold")
    a = tuple((m["role"], normalize(m["content"])) for m in left)
    b = tuple((m["role"], normalize(m["content"])) for m in right)
    if (
        not a
        or not b
        or not any(role == "user" and text for role, text in a)
        or not any(role == "user" and text for role, text in b)
    ):
        raise ValueError("Both prompts need a nonempty user message")
    if a == b:
        return {"candidate": True, "kind": "normalized_exact", "similarity": 1.0}
    if tuple(role for role, _ in a) != tuple(role for role, _ in b) or any(
        at != bt for (role, at), (_, bt) in zip(a, b) if role == "system"
    ):
        return {"candidate": False, "kind": "context_guard", "similarity": None}
    vectors = unit_vectors(np.array([left_vector, right_vector]))
    score = float(vectors[0] @ vectors[1])
    return {
        "candidate": score >= threshold,
        "kind": "embedding_cosine",
        "similarity": score,
    }
