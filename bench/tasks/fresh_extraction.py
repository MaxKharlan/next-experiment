"""Prepare an offline, reviewed extraction cohort before model evaluation.

This is a bounded synthetic task. Entity IDs, slot combinations and wording are
partitioned before generation. The text-only review parser is separate from the
generator; neither it nor hidden clean references enter the investigation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from pathlib import Path

from bench.faults.data import build as inject
from bench.tasks.structured import sha, write_json
from bench.tasks.structured_review import parse_request
from core.eval.structured import Ticket, validate_answer

SIZES = {"train": 432, "validation": 108, "check": 108, "test": 108}
ROLES = tuple(SIZES)
SEED = 2026101901


def semantic_key(value):
    due = value["due_date"]
    bucket = "none" if due is None else ("early" if int(due[-2:]) <= 13 else "late")
    return value["product"], value["priority"], value["callback"], bucket


def normalized(text):
    text = re.sub(r"T-\d{6}", "T-ID", text)
    return re.sub(r"2027-02-\d{2}", "DATE", text).lower()


def sample(role, i, groups, rng):
    product, priority, callback, bucket = groups[i % len(groups)]
    ticket = f"T-{600000 + ROLES.index(role) * 10000 + i}"
    due = (
        None
        if bucket == "none"
        else f"2027-02-{rng.randrange(1, 14) if bucket == 'early' else rng.randrange(14, 28):02d}"
    )
    clauses = [
        f"Ticket {ticket}; product: {product}.",
        f"The priority is {priority}.",
        "Please call me back." if callback else "Do not call me back.",
        f"The due date is {due}." if due else "No due date has been set.",
    ]
    # Distinct discourse order as well as distinct wrapper for every split.
    orders = ((1, 0, 3, 2), (2, 3, 0, 1), (3, 1, 2, 0), (0, 2, 1, 3))
    leads = (
        "Convert the service desk note to the requested record.",
        "Record the details from this incoming support note.",
        "Read this handover and produce the ticket fields.",
        "Create the structured ticket entry from this message.",
    )
    prompt = (
        leads[ROLES.index(role)]
        + " "
        + " ".join(clauses[j] for j in orders[ROLES.index(role)])
    )
    answer = {
        "ticket_id": ticket,
        "product": product,
        "priority": priority,
        "callback": callback,
        "due_date": due,
    }
    return {
        "messages": [
            {
                "role": "system",
                "content": "Extract ticket_id, product, priority, callback and due_date. Return only JSON matching this schema: "
                + json.dumps(Ticket.model_json_schema(), sort_keys=True),
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(answer, sort_keys=True)},
        ]
    }


def review(clean, historical):
    """Check references independently and fail on cross-split semantic leakage."""
    prompt_owners, entity_owners, group_owners = {}, {}, {}
    prior_prompts, prior_ids = set(), set()
    historical_hashes = {}
    for path in sorted(historical.rglob("*.jsonl")):
        historical_hashes[str(path)] = sha(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            text = next(m["content"] for m in row["messages"] if m["role"] == "user")
            prior_prompts.add(normalized(text))
            prior_ids.update(re.findall(r"T-\d{6}", text))
    counts = {}
    for role in ROLES:
        rows = [
            json.loads(line)
            for line in (clean / f"{role}.jsonl").read_text().splitlines()
        ]
        counts[role] = len(rows)
        if len(rows) != SIZES[role]:
            raise ValueError("Unexpected split size")
        for row in rows:
            text = row["messages"][1]["content"]
            value = validate_answer("json_extraction", row["messages"][-1]["content"])
            if value != parse_request("json_extraction", text):
                raise ValueError("Text-only oracle disagrees with reference")
            if normalized(text) in prior_prompts or value["ticket_id"] in prior_ids:
                raise ValueError("Historical prompt or entity reused")
            for key, owners in (
                (normalized(text), prompt_owners),
                (value["ticket_id"], entity_owners),
                (semantic_key(value), group_owners),
            ):
                if key in owners and (owners is entity_owners or owners[key] != role):
                    raise ValueError("Duplicate entity or cross-split semantic overlap")
                owners[key] = role
    return {
        "version": 1,
        "reference_review": "all rows, separate text-only rule parser",
        "human_review": False,
        "model_evaluation": False,
        "counts": counts,
        "historical_files_sha256": historical_hashes,
        "cross_split_semantic_groups": 0,
        "cross_split_normalized_prompts": 0,
        "historical_normalized_prompts": 0,
        "historical_entity_ids": 0,
        "semantic_group": ["product", "priority", "callback", "due_date_bucket"],
        "limitations": "Shared schema and bounded vocabulary. This measures held-out slot combinations, not general natural-language extraction. Historical semantic slot overlap is expected and is not claimed absent.",
        "reviewer_sha256": sha(Path("bench/tasks/structured_review.py")),
    }


def build(out, historical):
    if not historical.is_dir() or not list(historical.rglob("*.jsonl")):
        raise ValueError("Historical cohort is required for overlap review")
    out.mkdir(parents=True, exist_ok=False)
    clean = out / "evaluator-only" / "clean"
    clean.mkdir(parents=True)
    rng = random.Random(SEED)
    groups = list(
        itertools.product(
            ("router", "printer", "scanner"),
            ("low", "normal", "high"),
            (False, True),
            ("none", "early", "late"),
        )
    )
    rng.shuffle(groups)
    datasets = {}
    for position, (role, count) in enumerate(SIZES.items()):
        rows = [sample(role, i, groups[position::4], rng) for i in range(count)]
        path = clean / f"{role}.jsonl"
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        datasets[role] = {
            "ref": path.name,
            "format": "chat_jsonl",
            "examples": count,
            "sha256": sha(path),
        }
    manifest = {
        "version": 1,
        "task": "json_extraction",
        "seed": SEED,
        "datasets": datasets,
        "generator_sha256": sha(Path(__file__)),
        "license": "Apache-2.0",
        "provenance": "New authored synthetic templates; no provider calls",
        "evaluator": {
            "name": "ticket_json_v1",
            "metric": "json_fields_exact",
            "parameters": {"max_new_tokens": 192},
        },
    }
    write_json(clean / "manifest.json", manifest)
    reviewed = review(clean, historical)
    write_json(out / "review.json", reviewed)
    # Ground-truth injection manifests remain outside observed directories.
    for name, strength in (("mild", 0.1), ("strong", 0.3)):
        observed = out / "observed" / name
        observed.parent.mkdir(exist_ok=True)
        _, record = inject(
            clean, observed, strength, seed=SEED, keep=0.5, select="random"
        )
        (observed.parent / f"{name}.fault.json").unlink()
        write_json(out / "evaluator-only" / f"{name}.fault.json", record)
    write_json(
        out / "cohort.json",
        {
            "version": 1,
            "status": "data_reviewed_execution_not_frozen",
            "task": "json_extraction",
            "seed": SEED,
            "review_sha256": sha(out / "review.json"),
            "clean": manifest,
            "paid_jobs": 0,
            "final_test_model_evaluations": 0,
            "agent_access": "observed/<strength> train/validation only; evaluator owns check/test and evaluator-only",
            "required_before_submission": [
                "measured development noise and power/seed choice",
                "pinned runtime/model/tokenizer/template",
                "verified provider price and total cap",
                "execution protocol hash",
                "MPS resource handoff",
            ],
        },
    )
    return reviewed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--historical", type=Path, default=Path("datasets/structured-mixed-v1")
    )
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.historical), indent=2))


if __name__ == "__main__":
    main()
