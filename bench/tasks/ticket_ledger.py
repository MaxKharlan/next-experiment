"""Development-only ticket histories with explicit state dependencies.

This module never reads a previous cohort or a final-test split.
"""

from __future__ import annotations

import copy
import json
import random

FIELDS = ("product", "priority", "callback", "due_date")
VALUES = {
    "product": ("router", "printer", "scanner"),
    "priority": ("low", "normal", "high"),
    "callback": (False, True),
    "due_date": (None, "2031-06-11", "2031-06-19", "2031-06-27"),
}
SIZES = {"train": (72, 288), "validation": (24, 72), "check": (36, 144)}
SEED = 2026102007
ARMS = ("format", "healthy", "cut", "wrong")
FAMILIES = ("conditional", "restore", "copy")
SYSTEM = (
    "Read this ticket ledger in numbered order. SET assigns a field. COPY reads "
    "the source at that step. IF tests the current field before assigning. "
    "RESTORE reads the same ticket field immediately after the stated earlier "
    "step (step 0 means the initial records); it does not undo other fields. "
    "Return only the final requested ticket as JSON with exactly ticket_id, "
    "product, priority, callback and due_date. Values retain their JSON types."
)


def canonical(value):
    return json.dumps(value, sort_keys=True)


def execute(initial, events):
    """Generator oracle: forward snapshots of every ticket after every event."""
    state = copy.deepcopy(initial)
    snapshots = [copy.deepcopy(state)]
    for event in events:
        kind, ticket, field = event["kind"], event["ticket"], event["field"]
        if kind == "set":
            state[ticket][field] = event["value"]
        elif kind == "copy":
            state[ticket][field] = state[event["source"]][field]
        elif kind == "restore":
            state[ticket][field] = snapshots[event["after"]][ticket][field]
        elif kind == "if":
            if state[ticket][event["condition_field"]] == event["equals"]:
                state[ticket][field] = event["value"]
        else:
            raise ValueError(kind)
        snapshots.append(copy.deepcopy(state))
    return state


def render(initial, events, target):
    lines = [f"INITIAL {canonical(initial)}"]
    for step, event in enumerate(events, 1):
        ticket, field = event["ticket"], event["field"]
        if event["kind"] == "set":
            command = f"SET {ticket} {field} = {canonical(event['value'])}"
        elif event["kind"] == "copy":
            command = f"COPY {ticket} {field} FROM {event['source']}"
        elif event["kind"] == "restore":
            command = f"RESTORE {ticket} {field} AFTER {event['after']}"
        else:
            command = (
                f"IF {ticket} {event['condition_field']} = "
                f"{canonical(event['equals'])} THEN {field} = "
                f"{canonical(event['value'])}"
            )
        lines.append(f"{step}. {command}")
    return "\n".join([*lines, f"REPORT {target}"])


def sample(role, index, rng):
    family = FAMILIES[index % len(FAMILIES)]
    offset = {"train": 600000, "validation": 700000, "check": 800000}[role]
    ids = [f"T-{offset + 4 * index + i:06d}" for i in range(4)]
    initial = {
        ticket: {field: rng.choice(VALUES[field]) for field in FIELDS} for ticket in ids
    }
    target = rng.choice(ids)
    events = []
    # Three blocks, each with four distracting writes and two dependent events.
    # Random target, fields and values prevent fixed-position answer extraction.
    for _ in range(3):
        for _ in range(4):
            field = rng.choice(FIELDS)
            events.append(
                {
                    "kind": "set",
                    "ticket": rng.choice(ids),
                    "field": field,
                    "value": rng.choice(VALUES[field]),
                }
            )
        field = rng.choice(FIELDS)
        state = execute(initial, events)
        other = rng.choice([v for v in VALUES[field] if v != state[target][field]])
        if family == "conditional":
            condition = rng.choice([f for f in FIELDS if f != field])
            equals = rng.choice(VALUES[condition])
            events.append(
                {"kind": "set", "ticket": target, "field": field, "value": other}
            )
            events.append(
                {
                    "kind": "if",
                    "ticket": target,
                    "field": field,
                    "condition_field": condition,
                    "equals": equals,
                    "value": state[target][field],
                }
            )
        elif family == "restore":
            after = rng.randrange(len(events) + 1)
            events.append(
                {"kind": "set", "ticket": target, "field": field, "value": other}
            )
            events.append(
                {"kind": "restore", "ticket": target, "field": field, "after": after}
            )
        else:
            source = rng.choice([t for t in ids if t != target])
            events.append(
                {"kind": "copy", "ticket": target, "field": field, "source": source}
            )
            events.append(
                {"kind": "set", "ticket": source, "field": field, "value": other}
            )
    expected = {"ticket_id": target, **execute(initial, events)[target]}
    return {
        "id": f"ledger-v1-{role}-{index:04d}",
        "source": family,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": render(initial, events, target)},
            {"role": "assistant", "content": canonical(expected)},
        ],
    }


def direct(row):
    row = copy.deepcopy(row)
    row["source"] = "serialization"
    row["messages"][0]["content"] = (
        "Return the supplied object as JSON with the same fields and values."
    )
    row["messages"][1]["content"] = "Object: " + row["messages"][-1]["content"]
    return row


def datasets():
    rng = random.Random(SEED)
    result = {}
    injection = {}
    for role, (anchors, histories) in SIZES.items():
        rows = [sample(role, i, rng) for i in range(anchors + histories)]
        clean = [direct(r) for r in rows[:anchors]] + rows[anchors:]
        if role != "train":
            result[role] = clean
            continue
        selected = []
        for family in FAMILIES:
            candidates = [
                i for i in range(anchors, len(clean)) if clean[i]["source"] == family
            ]
            selected.extend(rng.sample(candidates, len(candidates) // 2))
        injection["train_indices_zero_based"] = sorted(selected)
        for arm in ARMS:
            modified = copy.deepcopy(clean)
            if arm == "format":
                modified = [direct(r) for r in modified]
            for i in selected:
                answer = modified[i]["messages"][-1]
                if arm == "cut":
                    answer["content"] = answer["content"][: len(answer["content"]) // 2]
                elif arm == "wrong":
                    value = json.loads(answer["content"])
                    for field in ("product", "priority"):
                        options = VALUES[field]
                        value[field] = options[(options.index(value[field]) + 1) % 3]
                    answer["content"] = canonical(value)
            result[f"{arm}-train"] = modified
    return result, injection
