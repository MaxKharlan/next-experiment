"""Deterministic faults 2–5 and observed-data repairs with explicit controls."""

from __future__ import annotations

import copy
import json
import math
import random

from core.probes.level0.overlap import normalize

FOREIGN_PREFIX = "[INST] "
FOREIGN_SUFFIX = " [/INST]"


def selection(rows, strength, seed):
    if not rows or not 0 < strength <= 1:
        raise ValueError("Require nonempty rows and strength in (0,1]")
    return sorted(
        random.Random(seed).sample(
            range(len(rows)), max(1, round(len(rows) * strength))
        )
    )


def template_mismatch(rows, strength, seed=0):
    changed = selection(rows, strength, seed)
    out = copy.deepcopy(rows)
    for index in changed:
        messages = out[index]["messages"]
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in messages[:-1])
        out[index]["messages"] = [
            {"role": "user", "content": FOREIGN_PREFIX + rendered + FOREIGN_SUFFIX},
            messages[-1],
        ]
    return out, changed


def contradictory_labels(rows, allowed, strength, seed=0):
    if len(set(allowed)) != len(allowed) or len(allowed) < 2:
        raise ValueError("Require at least two unique allowed labels")
    changed = selection(rows, strength, seed)
    out = copy.deepcopy(rows)
    rng = random.Random(seed + 1)
    for index in changed:
        original = out[index]["messages"][-1]["content"]
        if original not in allowed:
            raise ValueError("Observed label outside declared vocabulary")
        out[index]["messages"][-1]["content"] = rng.choice(
            [label for label in allowed if label != original]
        )
    return out, changed


def validation_overlap(train, validation, strength, seed=0):
    changed = selection(validation, strength, seed)
    if len(train) < len(changed):
        raise ValueError("Not enough distinct train rows")
    chosen = random.Random(seed + 1).sample(range(len(train)), len(changed))
    out = copy.deepcopy(validation)
    for target, source in zip(changed, chosen, strict=True):
        out[target] = copy.deepcopy(train[source])
    return out, {"validation_indices": changed, "training_indices": chosen}


def resample_label_distribution(rows, target, size, seed=0):
    if (
        type(size) is not int
        or size < 1
        or abs(sum(target.values()) - 1) > 1e-8
        or any(not math.isfinite(p) or p < 0 for p in target.values())
    ):
        raise ValueError("Invalid target distribution or size")
    groups = {
        label: [r for r in rows if r["messages"][-1]["content"] == label]
        for label in target
    }
    if any(target[label] > 0 and not group for label, group in groups.items()):
        raise ValueError("Cannot recover a missing class by resampling")
    labels = list(target)
    counts = {label: int(size * target[label]) for label in labels}
    residual = size - sum(counts.values())
    order = sorted(labels, key=lambda label: -(size * target[label] - counts[label]))
    for label in order[:residual]:
        counts[label] += 1
    rng = random.Random(seed)
    out = []
    for label in labels:
        out.extend(copy.deepcopy(rng.choices(groups[label], k=counts[label])))
    rng.shuffle(out)
    return out


def prompt_key(row):
    return json.dumps(
        [(m["role"], normalize(m["content"])) for m in row["messages"][:-1]]
    )


def overlap_lines(train, validation):
    keys = {prompt_key(row) for row in train}
    return [i for i, row in enumerate(validation) if prompt_key(row) in keys]


def drop_candidates_with_control(rows, indices, seed=0):
    flagged = set(indices)
    if (
        not flagged
        or any(type(i) is not int or not 0 <= i < len(rows) for i in flagged)
        or len(flagged) >= len(rows)
    ):
        raise ValueError("Need a proper nonempty set of observed candidate indices")
    control = set(random.Random(seed).sample(range(len(rows)), len(flagged)))
    return (
        [copy.deepcopy(row) for i, row in enumerate(rows) if i not in flagged],
        [copy.deepcopy(row) for i, row in enumerate(rows) if i not in control],
    )


def template_evidence(rows):
    indices = [
        i
        for i, row in enumerate(rows)
        if len(row["messages"]) == 2
        and row["messages"][0]["role"] == "user"
        and row["messages"][0]["content"].startswith(FOREIGN_PREFIX)
        and row["messages"][0]["content"].endswith(FOREIGN_SUFFIX)
    ]
    return {
        "candidate_indices": indices,
        "scope": "Exact supported foreign wrapper; legitimate quoted syntax can match. Repair requires the declared producer format; no automatic stripping.",
    }
