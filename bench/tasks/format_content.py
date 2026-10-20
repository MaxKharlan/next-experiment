"""New development histories with a shared serialization anchor and four arms."""

from __future__ import annotations

import copy
import itertools
import json
import random
import re

from bench.faults.data import cut
from bench.tasks.structured import sha, write_json
from bench.tasks.ticket_updates import parse_history, sample
from core.eval.structured import Ticket, validate_answer

SIZES = {"train": (72, 144), "validation": (12, 24), "check": (24, 48)}
ARMS = ("format", "healthy", "cut", "wrong")


def direct(row):
    result = copy.deepcopy(row)
    value = json.loads(row["messages"][-1]["content"])
    result["source"] = "serialization"
    result["messages"][0]["content"] = (
        "Serialize the supplied field values without changing them. Return only JSON matching this schema: "
        + json.dumps(Ticket.model_json_schema(), sort_keys=True)
    )
    result["messages"][1]["content"] = "Fields to serialize:\n" + "\n".join(
        f"{key}={json.dumps(value[key])}" for key in sorted(value)
    )
    parsed = {
        key: json.loads(raw)
        for key, raw in (
            line.split("=", 1)
            for line in result["messages"][1]["content"].splitlines()[1:]
        )
    }
    if parsed != json.loads(result["messages"][-1]["content"]):
        raise ValueError("Direct serialization reference mismatch")
    return result


def make_row(role, index, rng):
    row = sample(role, index, rng)
    # The old generator's data are development-only; new seed, IDs and dates here.
    for message in row["messages"]:
        message["content"] = re.sub(
            r"T-(\d{6})", lambda m: f"T-{int(m[1]) + 100000}", message["content"]
        )
        message["content"] = message["content"].replace("2028-03", "2029-04")
    return row


def canonical(value):
    return json.dumps(value, sort_keys=True)


def alternatives(expected, tokenizer, rng):
    value = validate_answer("json_extraction", expected)
    size = len(tokenizer.encode(expected, add_special_tokens=False))
    wrong = []
    for product, priority in itertools.product(
        ("router", "printer", "scanner"), ("low", "normal", "high")
    ):
        candidate = canonical({**value, "product": product, "priority": priority})
        if (
            candidate != expected
            and len(tokenizer.encode(candidate, add_special_tokens=False)) == size
        ):
            wrong.append(candidate)
    if len(wrong) < 3:
        raise ValueError("Need three wrong equal-token-length choices")
    choices = [expected, *rng.sample(wrong, 3)]
    rng.shuffle(choices)
    return {
        "choices": choices,
        "correct_index": choices.index(expected),
        "tokens_per_choice": size,
    }


def build(root, tokenizer):
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(2026101903)
    manifest, seen_ids, seen_histories = {}, set(), set()
    for role, (n_format, n_content) in SIZES.items():
        histories = [make_row(role, i, rng) for i in range(n_format + n_content)]
        for row in histories:
            text, expected = (
                row["messages"][1]["content"],
                row["messages"][-1]["content"],
            )
            if parse_history(text) != json.loads(expected):
                raise ValueError("Reference review failed")
            ids = list(dict.fromkeys(re.findall(r"T-\d{6}", text)))
            if seen_ids.intersection(ids):
                raise ValueError("Reused entity")
            seen_ids.update(ids)
            normalized = text
            for i, entity in enumerate(ids):
                normalized = normalized.replace(entity, f"T-ENTITY-{i}")
            if normalized in seen_histories:
                raise ValueError("Normalized history overlap")
            seen_histories.add(normalized)
        anchors = [direct(r) for r in histories[:n_format]]
        content = histories[n_format:]
        if role == "check":
            for kind, rows in (("format", anchors), ("content", content)):
                path = root / f"check-{kind}.jsonl"
                path.write_text(
                    "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
                )
                manifest[f"check_{kind}"] = {
                    "path": str(path.resolve()),
                    "sha256": sha(path),
                    "examples": len(rows),
                }
            choices = [
                alternatives(r["messages"][-1]["content"], tokenizer, rng)
                for r in content
            ]
            write_json(root / "choices.json", choices)
        else:
            # Exactly half of history answers in each family; anchors remain clean.
            selected = []
            for family in sorted({r["source"] for r in content}):
                indices = [i for i, r in enumerate(content) if r["source"] == family]
                selected.extend(rng.sample(indices, len(indices) // 2))
            write_json(
                root / f"{role}-injection.json",
                sorted(i + n_format + 1 for i in selected),
            )
            for arm in ARMS:
                rows = copy.deepcopy(anchors + content)
                if arm == "format":
                    rows = [direct(r) for r in rows]
                for i in selected:
                    row = rows[n_format + i]
                    if arm == "cut":
                        row["messages"][-1]["content"] = cut(
                            row["messages"][-1]["content"], 0.5
                        )
                    elif arm == "wrong":
                        value = json.loads(row["messages"][-1]["content"])
                        # Systematic cyclic wrong values, same schema and length family.
                        for key, options in (
                            ("product", ("router", "printer", "scanner")),
                            ("priority", ("low", "normal", "high")),
                        ):
                            value[key] = options[(options.index(value[key]) + 1) % 3]
                        row["messages"][-1]["content"] = canonical(value)
                path = root / f"{arm}-{role}.jsonl"
                path.write_text(
                    "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
                )
                manifest[f"{arm}_{role}"] = {
                    "path": str(path.resolve()),
                    "sha256": sha(path),
                    "examples": len(rows),
                }
    write_json(
        root / "review.json",
        {
            "reference_review": "All clean histories checked by text-only replay parser; direct anchors checked against their explicit field values. One author; no independent human review.",
            "unique_entities": len(seen_ids),
            "unique_normalized_histories": len(seen_histories),
            "sizes": SIZES,
            "arms": list(ARMS),
            "data": manifest,
            "corruption": "50% of content rows per family (1/3 of total training rows); same selected rows for cut and wrong; shared format anchors are clean.",
            "scope": "New development train/validation/check only, no final test. Grammar shared across splits. Format-only arm has different input lengths; not a compute-matched causal control.",
        },
    )
    return manifest
