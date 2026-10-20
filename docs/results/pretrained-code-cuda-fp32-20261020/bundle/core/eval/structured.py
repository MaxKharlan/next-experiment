"""Strict contracts and metrics for the versioned synthetic A2/A3 tasks."""

from __future__ import annotations

import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Ticket(StrictRecord):
    ticket_id: str = Field(pattern=r"^T-[0-9]{6}$")
    product: Literal["router", "printer", "scanner"]
    priority: Literal["low", "normal", "high"]
    callback: StrictBool
    due_date: str | None

    @field_validator("due_date")
    @classmethod
    def valid_date(cls, value):
        if value is not None and date.fromisoformat(value).isoformat() != value:
            raise ValueError("Use an ISO calendar date")
        return value


class TrackOrder(StrictRecord):
    order_id: str = Field(pattern=r"^O-[0-9]{6}$")


class CancelOrder(TrackOrder):
    reason: Literal["duplicate", "changed_mind", "wrong_item"]


class SchedulePickup(TrackOrder):
    date: str
    location: Literal["front_desk", "loading_bay", "locker"]

    @field_validator("date")
    @classmethod
    def valid_date(cls, value):
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("Use an ISO calendar date")
        return value


FUNCTIONS = {
    "track_order": (TrackOrder, "Get the delivery status of an order."),
    "cancel_order": (CancelOrder, "Cancel an order for the specified reason."),
    "schedule_pickup": (SchedulePickup, "Book an order pickup at a date and location."),
}


class FunctionCall(StrictRecord):
    name: Literal["track_order", "cancel_order", "schedule_pickup"]
    arguments: dict


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def validate_answer(task, text):
    value = strict_json(text)
    if task == "json_extraction":
        return Ticket.model_validate(value).model_dump()
    if task == "function_calling":
        call = FunctionCall.model_validate(value)
        args = FUNCTIONS[call.name][0].model_validate(call.arguments).model_dump()
        return {"name": call.name, "arguments": args}
    raise ValueError(f"Unknown structured task: {task}")


def json_valid(answer, expected):
    try:
        strict_json(answer)
        return 1.0
    except (ValueError, TypeError, RecursionError):
        return 0.0


def structured_exact(task, answer, expected):
    reference = validate_answer(task, expected)
    try:
        return float(validate_answer(task, answer) == reference)
    except (ValueError, TypeError, RecursionError):
        return 0.0


def json_fields_exact(answer, expected):
    return structured_exact("json_extraction", answer, expected)


def function_call_exact(answer, expected):
    return structured_exact("function_calling", answer, expected)


def structured_summary(task, answers, expected):
    if not answers or len(answers) != len(expected):
        raise ValueError("Require aligned, nonempty predictions and references")
    rows = []
    for answer, reference in zip(answers, expected):
        truth = validate_answer(task, reference)
        row = {
            "valid_json": json_valid(answer, reference),
            "schema_valid": 0.0,
            "exact_match": 0.0,
        }
        try:
            value = validate_answer(task, answer)
            row.update(schema_valid=1.0, exact_match=float(value == truth))
            if task == "json_extraction":
                row["field_accuracy"] = sum(value[k] == truth[k] for k in truth) / len(
                    truth
                )
            else:
                row["name_accuracy"] = float(value["name"] == truth["name"])
        except (ValueError, TypeError, RecursionError):
            pass
        rows.append(row)
    keys = {
        "valid_json",
        "schema_valid",
        "exact_match",
        "field_accuracy" if task == "json_extraction" else "name_accuracy",
    }
    return {
        key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in sorted(keys)
    }


TASK_EVALUATORS = {
    "banking77": ("intent_label", "exact_match", 32),
    "json_extraction": ("ticket_json_v1", "json_fields_exact", 192),
    "function_calling": ("order_tools_v1", "function_call_exact", 192),
}


def validate_evaluator(task, spec):
    expected = TASK_EVALUATORS.get(task)
    if expected is None or (spec.name, spec.metric) != expected[:2]:
        raise ValueError("Unsupported task/evaluator contract")
    if not spec.higher_is_better or set(spec.parameters) - {"max_new_tokens"}:
        raise ValueError("Unsupported evaluator parameters")
    limit = spec.parameters.get("max_new_tokens", expected[2])
    if type(limit) is not int or not 1 <= limit <= 1024:
        raise ValueError("max_new_tokens must be an integer from 1 to 1024")
    return spec.model_copy(update={"parameters": {"max_new_tokens": limit}})
