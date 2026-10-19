"""Independent rule check for the bounded synthetic fixture vocabulary.

This parser reads user text only. It is an integrity oracle for this dataset,
not a learned model or a claim about extraction on arbitrary natural language.
"""

import re


def only(pattern, text):
    values = set(re.findall(pattern, text, flags=re.IGNORECASE))
    if len(values) != 1:
        raise ValueError(f"Expected exactly one value for {pattern}")
    return next(iter(values))


def calendar_date(text):
    dates = re.findall(r"2027-02-\d{2}", text)
    dates += [
        f"2027-02-{int(day):02d}"
        for day in re.findall(r"February (\d{1,2}), 2027", text)
    ]
    if len(set(dates)) != 1:
        raise ValueError("Expected exactly one date")
    return dates[0]


def parse_request(task, text):
    lower = text.lower()
    if task == "json_extraction":
        priorities = re.findall(r"priority (?:is )?(low|normal|high)", lower)
        priorities += re.findall(r"(low|normal|high) priority", lower)
        if len(set(priorities)) != 1:
            raise ValueError("Ambiguous priority")
        negative = bool(
            re.search(r"do not (?:want a callback|call(?: me)? back)", lower)
        )
        positive = bool(
            re.search(r"please call(?: me)? back|would like a callback", lower)
        )
        if negative == positive:
            raise ValueError("Ambiguous callback")
        return {
            "ticket_id": only(r"T-\d{6}", text),
            "product": only(r"\b(router|printer|scanner)\b", lower),
            "priority": priorities[0],
            "callback": positive,
            "due_date": None if "no due date" in lower else calendar_date(text),
        }
    if task != "function_calling":
        raise ValueError("Unknown task")
    args = {"order_id": only(r"O-\d{6}", text)}
    choices = [
        ("track_order", "delivery status" in lower),
        ("cancel_order", "cancel" in lower),
        ("schedule_pickup", "pickup" in lower),
    ]
    names = [name for name, matches in choices if matches]
    if len(names) != 1:
        raise ValueError("Ambiguous function intent")
    name = names[0]
    if name == "cancel_order":
        normalized = lower.replace("changed my mind", "changed_mind")
        args["reason"] = only(r"duplicate|changed_mind|wrong_item", normalized)
    elif name == "schedule_pickup":
        normalized = lower.replace("front desk", "front_desk").replace(
            "loading bay", "loading_bay"
        )
        args.update(
            date=calendar_date(text),
            location=only(r"front_desk|loading_bay|locker", normalized),
        )
    return {"name": name, "arguments": args}
