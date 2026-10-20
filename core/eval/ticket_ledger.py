"""Text-only review oracle, separate from the forward data generator.

Resolve each requested field backwards through the ledger. No generator event
objects or labels are consumed. Both implementations still share one author.
"""

from __future__ import annotations

import json
import re
from functools import cache

from core.eval.structured import Ticket


def parse(text):
    lines = text.splitlines()
    if len(lines) < 3 or not lines[0].startswith("INITIAL "):
        raise ValueError("Missing initial ledger")
    initial = json.loads(lines[0][8:])
    if not re.fullmatch(r"REPORT T-\d{6}", lines[-1]):
        raise ValueError("Missing report target")
    target = lines[-1].split()[1]
    operations = []
    for step, line in enumerate(lines[1:-1], 1):
        prefix = f"{step}. "
        if not line.startswith(prefix):
            raise ValueError("Nonconsecutive step")
        command = line[len(prefix) :]
        patterns = (
            ("set", r"SET (T-\d{6}) (\w+) = (.+)"),
            ("copy", r"COPY (T-\d{6}) (\w+) FROM (T-\d{6})"),
            ("restore", r"RESTORE (T-\d{6}) (\w+) AFTER (\d+)"),
            ("if", r"IF (T-\d{6}) (\w+) = (.+) THEN (\w+) = (.+)"),
        )
        for kind, pattern in patterns:
            match = re.fullmatch(pattern, command)
            if match:
                operations.append((kind, match.groups()))
                break
        else:
            raise ValueError("Unknown command")
        if kind == "restore" and not 0 <= int(match[3]) < step:
            raise ValueError("Restore must refer to an earlier step")
    return initial, operations, target


def review(text, *, ignore_conditions=False, ignore_restore=False, late_copy=False):
    initial, operations, target = parse(text)

    @cache
    def resolve(ticket, field, through):
        for i in range(through, 0, -1):
            kind, args = operations[i - 1]
            destination, changed = args[0], args[3] if kind == "if" else args[1]
            if destination != ticket or changed != field:
                continue
            if kind == "set":
                return json.loads(args[2])
            if kind == "copy":
                # A bounded shortcut: read source after the immediately next
                # event instead of at copy time. Never follow a future copy.
                source_time = i - 1
                if late_copy and i < len(operations):
                    next_kind, next_args = operations[i]
                    if next_kind == "set" and next_args[:2] == (args[2], field):
                        return json.loads(next_args[2])
                return resolve(args[2], field, source_time)
            if kind == "restore":
                if ignore_restore:
                    continue
                return resolve(ticket, field, int(args[2]))
            if ignore_conditions or resolve(ticket, args[1], i - 1) == json.loads(
                args[2]
            ):
                return json.loads(args[4])
        return initial[ticket][field]

    return Ticket.model_validate(
        {
            "ticket_id": target,
            **{
                field: resolve(target, field, len(operations))
                for field in ("product", "priority", "callback", "due_date")
            },
        }
    ).model_dump()
