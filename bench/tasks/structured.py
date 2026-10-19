"""Build deterministic, synthetic A2/A3 task bundles; no remote service calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from core.case import InvestigationCase
from core.eval.data import read_examples
from core.eval.structured import FUNCTIONS, TASK_EVALUATORS, Ticket, validate_answer

SIZES = {"train": 384, "validation": 96, "check": 96, "test": 96}
ROLES = tuple(SIZES)
# Different wording per split; entity IDs never cross split boundaries.
WRAPPERS = {
    "train": (
        "Please process this request: {}",
        "Request details: {}",
        "I need help. {}",
        "Here is my request. {}",
    ),
    "validation": (
        "Can you handle the following? {}",
        "For your attention: {}",
        "Please act on these details. {}",
        "My request is as follows: {}",
    ),
    "check": (
        "New request to process: {}",
        "Could you assist with this? {}",
        "Please use the following information: {}",
        "I am contacting you about this: {}",
    ),
    "test": (
        "Handle this request using the provided details: {}",
        "Please take care of this. {}",
        "This is what I need: {}",
        "The following request needs processing: {}",
    ),
}


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool_schema():
    return [
        {
            "name": name,
            "description": description,
            "parameters": cls.model_json_schema(),
        }
        for name, (cls, description) in FUNCTIONS.items()
    ]


def sample(task, index, role, rng):
    serial = (ROLES.index(role) + 1) * 100000 + index
    day = f"2027-02-{rng.randrange(1, 28):02d}"
    if task == "json_extraction":
        answer = {
            "ticket_id": f"T-{serial}",
            "product": ("router", "printer", "scanner")[index % 3],
            "priority": ("low", "normal", "high")[(index // 3) % 3],
            "callback": bool(index % 2),
            "due_date": day if index % 4 else None,
        }
        text = (
            f"Ticket {answer['ticket_id']} concerns a {answer['product']}. "
            f"The priority is {answer['priority']}. "
            + (
                "Please call me back. "
                if answer["callback"]
                else "Do not call me back. "
            )
            + (
                f"The due date is {day}."
                if answer["due_date"]
                else "No due date has been set."
            )
        )
        instruction = (
            "Extract ticket_id, product, priority, callback and due_date. Return only JSON matching this schema: "
            + json.dumps(Ticket.model_json_schema(), sort_keys=True)
        )
    else:
        name = tuple(FUNCTIONS)[index % 3]
        args = {"order_id": f"O-{serial}"}
        if name == "track_order":
            text = f"Tell me the delivery status for order {args['order_id']}."
        elif name == "cancel_order":
            args["reason"] = ("duplicate", "changed_mind", "wrong_item")[
                (index // 3) % 3
            ]
            text = f"Cancel order {args['order_id']}. Reason: {args['reason']}."
        else:
            args.update(
                date=day,
                location=("front_desk", "loading_bay", "locker")[(index // 3) % 3],
            )
            text = f"Arrange pickup of order {args['order_id']} on {day} at {args['location']}."
        answer = {"name": name, "arguments": args}
        instruction = (
            "Choose exactly one function. Return only a JSON object with name and arguments. Available functions: "
            + json.dumps(tool_schema(), sort_keys=True)
        )
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": WRAPPERS[role][index % 4].format(text)},
            {"role": "assistant", "content": json.dumps(answer, sort_keys=True)},
        ]
    }


def validate_bundle(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha(directory / "schema.json") != manifest["schema_sha256"]:
        raise ValueError("Schema hash mismatch")
    seen_prompts, seen_ids = set(), set()
    for role, spec in manifest["datasets"].items():
        path = directory / spec["ref"]
        if sha(path) != spec["sha256"]:
            raise ValueError(f"Hash mismatch: {role}")
        rows = read_examples(path)
        if len(rows) != spec["examples"]:
            raise ValueError(f"Count mismatch: {role}")
        for row in rows:
            value = validate_answer(manifest["task"], row.expected)
            prompt = json.dumps(row.prompt, sort_keys=True)
            identity = (
                value["ticket_id"]
                if manifest["task"] == "json_extraction"
                else value["arguments"]["order_id"]
            )
            if prompt in seen_prompts or identity in seen_ids:
                raise ValueError("Duplicate prompt or entity across bundle")
            seen_prompts.add(prompt)
            seen_ids.add(identity)
    return {"task": manifest["task"], "examples": len(seen_ids), "valid": True}


def build(out, seed=20261019, super_data=None):
    out.mkdir(parents=True, exist_ok=False)
    augmentation = json.loads(Path(super_data).read_text()) if super_data else None
    for task in ("json_extraction", "function_calling"):
        directory = out / task
        directory.mkdir()
        rng = random.Random(seed)
        datasets = {}
        for role, count in SIZES.items():
            path = directory / f"{role}.jsonl"
            rows = [sample(task, i, role, rng) for i in range(count)]
            if augmentation:
                batches = [
                    b
                    for b in augmentation["batches"]
                    if b["task"] == task and b["split"] == role
                ]
                if len(batches) != 1 or len(batches[0]["rows"]) != 6:
                    raise ValueError(
                        "Require one six-row Super batch per task and split"
                    )
                rows.extend(batches[0]["rows"])
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            datasets[role] = {
                "ref": path.name,
                "format": "chat_jsonl",
                "examples": len(rows),
                "sha256": sha(path),
            }
        name, metric, limit = TASK_EVALUATORS[task]
        evaluator = {
            "name": name,
            "metric": metric,
            "parameters": {"max_new_tokens": limit},
        }
        manifest = {
            "version": 1,
            "task": task,
            "seed": seed,
            "datasets": datasets,
            "evaluator": evaluator,
            "license": "Apache-2.0",
            "provenance": "Authored templates and synthetic slots"
            + (" plus 24 recorded Super paraphrases" if augmentation else ""),
            "super_source_sha256": sha(Path(super_data)) if augmentation else None,
            "super_model": augmentation["model"] if augmentation else None,
            "source_counts": {
                role: {"local": count, "super": 6 if augmentation else 0}
                for role, count in SIZES.items()
            },
            "human_review": "not performed; user requested agent review",
            "split_policy": "Distinct entity IDs and outer wording; shared semantic templates",
            "generator_sha256": sha(Path(__file__)),
        }
        write_json(directory / "manifest.json", manifest)
        write_json(
            directory / "schema.json",
            Ticket.model_json_schema() if task == "json_extraction" else tool_schema(),
        )
        manifest["schema_sha256"] = sha(directory / "schema.json")
        write_json(directory / "manifest.json", manifest)
        # This is a real dataset-only audit case, not a fictional provider job.
        case = InvestigationCase.model_validate(
            {
                "task": task,
                "mode": "B",
                "symptom": "Inspect this structured task dataset before training",
                "job": {
                    "id": "local:dataset-audit",
                    "model": "not-trained",
                    "hyperparameters": {},
                },
                "datasets": {
                    role: {**spec, "ref": str(directory / spec["ref"])}
                    for role, spec in datasets.items()
                },
                "evaluator": evaluator,
                "budget_usd": 5.0,
            }
        )
        # Resolve relative refs against this file's directory with load_case().
        portable = case.model_dump()
        for role, spec in datasets.items():
            portable["datasets"][role]["ref"] = spec["ref"]
        write_json(directory / "audit-case.json", portable)
        write_json(directory / "validation.json", validate_bundle(directory))


def load_case(path):
    path = Path(path)
    value = json.loads(path.read_text())
    for spec in value["datasets"].values():
        ref = Path(spec["ref"])
        spec["ref"] = str(ref if ref.is_absolute() else (path.parent / ref).resolve())
    return InvestigationCase.model_validate(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--super-data", type=Path)
    args = parser.parse_args()
    build(args.out, args.seed, args.super_data)


if __name__ == "__main__":
    main()
