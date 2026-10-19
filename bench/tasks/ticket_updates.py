"""Development-only ticket histories with distractors and void updates.

No final test is created. A separate text parser reviews every reference.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from bench.tasks.structured import sha, write_json
from core.eval.structured import Ticket, validate_answer

SIZES = {"train": 192, "validation": 48, "check": 72}
FAMILIES = ("correction", "distractor", "void_updates")
FIELDS = ("product", "priority", "callback", "due_date")


def values(rng):
    return {
        "product": rng.choice(("router", "printer", "scanner")),
        "priority": rng.choice(("low", "normal", "high")),
        "callback": rng.choice((False, True)),
        "due_date": rng.choice((None, "2028-03-04", "2028-03-17", "2028-03-26")),
    }


def parse_history(text):
    """Parse only the input text, ignoring the generator's state and labels."""
    lines = text.splitlines()
    match = re.fullmatch(r"Return the current record for (T-\d{6})\.", lines[0])
    if not match:
        raise ValueError("Missing target")
    target = match[1]
    records = {}
    for line in lines[1:]:
        initial = re.fullmatch(r"Initial (T-\d{6}): (.+)", line)
        update = re.fullmatch(
            r"Update (\d+) for (T-\d{6}) \[(applied|void)\]: (\w+)=(.+)", line
        )
        if initial:
            if initial[1] in records:
                raise ValueError("Duplicate initial record")
            fields = dict(part.split("=", 1) for part in initial[2].split("; "))
            if set(fields) != set(FIELDS):
                raise ValueError("Incomplete initial fields")
            records[initial[1]] = {k: json.loads(v) for k, v in fields.items()}
        elif update:
            if update[2] not in records or update[4] not in FIELDS:
                raise ValueError("Unknown update target or field")
            if update[3] == "applied":
                records[update[2]][update[4]] = json.loads(update[5])
        else:
            raise ValueError("Unknown history syntax")
    return Ticket.model_validate({"ticket_id": target, **records[target]}).model_dump()


def sample(role, index, rng):
    family = FAMILIES[index % 3]
    offset = list(SIZES).index(role) * 10000
    target, other = (f"T-{800000 + offset + index * 2 + i}" for i in range(2))
    target_state, other_state = values(rng), values(rng)
    text = [f"Return the current record for {target}."]
    for ticket, state in ((target, target_state), (other, other_state)):
        text.append(
            f"Initial {ticket}: "
            + "; ".join(f"{k}={json.dumps(v)}" for k, v in state.items())
        )
    expected = dict(target_state)
    events = []
    for field in rng.sample(list(FIELDS), len(FIELDS)):
        replacement = values(rng)[field]
        events.append((target, "applied", field, replacement))
        expected[field] = replacement
        if family != "correction":
            events.append((other, "applied", field, values(rng)[field]))
        if family == "void_updates":
            # Deliberately conflicting ignored edits, always after the real edit.
            choices = {
                "product": ["router", "printer", "scanner"],
                "priority": ["low", "normal", "high"],
                "callback": [True, False],
                "due_date": [None, "2028-03-04", "2028-03-17"],
            }[field]
            ignored = next(v for v in choices if v != replacement)
            events.append((target, "void", field, ignored))
    for number, (ticket, status, field, value) in enumerate(events, 1):
        text.append(
            f"Update {number} for {ticket} [{status}]: {field}={json.dumps(value)}"
        )
    instruction = (
        "Read the ticket history in listed order. Start from the initial record, then apply only updates marked [applied] for the requested ticket. "
        "Ignore every [void] update and all other tickets. For each field the last applied value wins. Return only JSON matching this schema: "
        + json.dumps(Ticket.model_json_schema(), sort_keys=True)
    )
    return {
        "source": family,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "\n".join(text)},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"ticket_id": target, **expected}, sort_keys=True
                ),
            },
        ],
    }


def build(root):
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(2026101902)
    manifest, seen = {}, set()
    for role, count in SIZES.items():
        rows = [sample(role, index, rng) for index in range(count)]
        for row in rows:
            text = row["messages"][1]["content"]
            expected = validate_answer(
                "json_extraction", row["messages"][-1]["content"]
            )
            if parse_history(text) != expected:
                raise ValueError("Text review disagrees")
            ids = set(re.findall(r"T-\d{6}", text))
            if ids & seen:
                raise ValueError("Entity reused")
            seen.update(ids)
        path = root / f"{role}.jsonl"
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        manifest[role] = {
            "path": str(path.resolve()),
            "sha256": sha(path),
            "examples": len(rows),
        }
    write_json(
        root / "review.json",
        {
            "datasets": manifest,
            "reference_disagreements": 0,
            "entity_overlap": 0,
            "families": list(FAMILIES),
            "review": "All references checked by a separate text-only parser; generator and review authored by one session; no human review.",
            "split_scope": "Development train/validation/check; shared grammar and slot vocabulary, unique histories and entity IDs. No claim of unseen grammar generalization. No final test created or read.",
            "generator_sha256": sha(Path(__file__)),
        },
    )
    return manifest
