"""Prepare or verify a prospective development bundle. No network/model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from bench.tasks.ticket_ledger import ARMS, FAMILIES, SEED, SIZES, datasets
from core.eval.structured import validate_answer
from core.eval.ticket_ledger import review

SOURCES = (
    "bench/tasks/ticket_ledger.py",
    "core/eval/ticket_ledger.py",
    "core/eval/structured.py",
    "scripts/prepare_ticket_ledger.py",
)
PROTOCOL = {
    "id": "ticket-ledger-development-20261020-v2",
    "phase": "development_preparation",
    "data_seed": SEED,
    "order_seeds": [2401, 2402, 2403],
    "model": "Qwen/Qwen3-0.6B",
    "arms": list(ARMS),
    "recipe": {
        "n_epochs": 2,
        "learning_rate": 0.0001,
        "batch_size": 8,
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "context_length": 8192,
        "packing": False,
    },
    "generation": {
        "do_sample": False,
        "enable_thinking": False,
        "max_new_tokens": 192,
    },
    "metrics": {
        "primary": "Strict full-object exact on every check history; invalid, missing and limit-hit answers score zero.",
        "format": "Strict object exact on all direct check anchors.",
        "secondary": "Schema validity; all-denominator field accuracy; per-family strict exact; EOS/limit hits.",
        "unit": "Training seed. Paired arms share data, order seed and check rows. Rows are not independent training replicates.",
    },
    "gates": {
        "format_and_healthy_anchor_exact_min": 0.90,
        "healthy_history_exact_min": 0.40,
        "healthy_history_exact_max": 0.95,
        "healthy_minus_format_history_exact_min": 0.15,
        "healthy_minus_wrong_history_exact_min": 0.10,
        "healthy_each_family_exact_min": 0.25,
        "distinct_weight_digests_per_arm": 3,
    },
    "gate_aggregation": "Arithmetic mean of three prespecified seeds per arm; family floor uses seed mean within each family. All gates must pass together. No rounding before comparison.",
    "stopping": "Any failed gate terminates this design. No extra seeds, altered thresholds, endpoint substitution or infrastructure retry without a separately reviewed authorization. Missing jobs/scores are incomplete, never passes.",
    "power": "Not selected or computed here. Only after all admission gates pass may a separately frozen noise/power plan be prepared. Nonzero observed SD alone is insufficient.",
    "validation": "Common clean validation for every arm. Fixed epochs; no recipe selection or early stopping on check.",
    "corruption": "Exactly half of train histories within each family, same indices for cut and wrong. Cut retains first half of characters; wrong cycles product/priority with valid JSON. Anchors stay clean.",
    "controls": "Format-only arm serializes explicit answers and has shorter inputs; it is a learning diagnostic, not a compute-matched causal repair control.",
    "preregistration_scope": "Prospective data/endpoint/recipe lock before model measurements. No empirical admission result. Synthetic grammar shared across splits; no final-test data generated or read.",
    "execution_authorized": False,
    "execution_dependencies": [
        "Freeze tokenizer/model revisions, runtime and executor source digests before any model measurement.",
        "Count rendered tokens locally and reject any over-context row; do not truncate.",
        "Review data limitations and obtain an explicit new paid budget approval.",
        "Reconcile or retain previous ledger holds; this bundle neither copies nor changes a spending ledger.",
        "Create per-attempt reservations and immutable receipts before an authorized submission.",
    ],
    "model_measurements": None,
    "admission": "not_measured",
    "deployment": "T=1; classifiers offline.",
}


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(data):
    seen_ids, seen_histories = set(), set()
    counts = {}
    shortcut = {}
    for split in ("healthy-train", "validation", "check"):
        rows = data[split]
        counts[split] = {"anchors": 0, **dict.fromkeys(FAMILIES, 0)}
        shortcuts = {name: {"correct": 0, "total": 0} for name in FAMILIES}
        for row in rows:
            expected = validate_answer(
                "json_extraction", row["messages"][-1]["content"]
            )
            text = row["messages"][1]["content"]
            ids = list(dict.fromkeys(re.findall(r"T-\d{6}", text)))
            if seen_ids.intersection(ids):
                raise ValueError("Entity overlap")
            seen_ids.update(ids)
            if row["source"] == "serialization":
                if json.loads(text.removeprefix("Object: ")) != expected:
                    raise ValueError("Anchor mismatch")
                counts[split]["anchors"] += 1
                continue
            normalized = text
            for i, ticket in enumerate(ids):
                normalized = normalized.replace(ticket, f"ENTITY-{i}")
            if normalized in seen_histories:
                raise ValueError("Normalized history overlap")
            seen_histories.add(normalized)
            family = row["source"]
            counts[split][family] += 1
            if review(text) != expected:
                raise ValueError(f"Oracle disagreement: {row['id']}")
            options = {
                "conditional": {"ignore_conditions": True},
                "restore": {"ignore_restore": True},
                "copy": {"late_copy": True},
            }
            shortcuts[family]["correct"] += int(
                review(text, **options[family]) == expected
            )
            shortcuts[family]["total"] += 1
        shortcut[split] = shortcuts
    return {
        "status": "offline_checks_passed",
        "model_admission": "not_measured",
        "counts": counts,
        "unique_entities": len(seen_ids),
        "unique_normalized_histories": len(seen_histories),
        "shortcut_exact_counts": shortcut,
        "review": "Forward event oracle versus backwards text-only field resolver; one author, no independent human annotation.",
        "limitations": "Shortcut scores describe three deterministic mistakes, not model accuracy or causal specificity. Entity-disjoint splits share grammar; serialization anchors can share normalized field combinations. No historical final test was read.",
        "service_spend_usd": 0,
    }


def bundle():
    data, injection = datasets()
    files = {
        f"{name}.jsonl": b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows
        )
        for name, rows in data.items()
    }
    files["injection.json"] = encode(injection)
    files["audit.json"] = encode(audit(data))
    files["protocol.json"] = encode(
        {
            **PROTOCOL,
            "sizes": SIZES,
            "files_sha256": {name: digest(value) for name, value in files.items()},
            "source_sha256": {
                name: digest(Path(name).read_bytes()) for name in SOURCES
            },
        }
    )
    files["protocol.sha256"] = (digest(files["protocol.json"]) + "\n").encode()
    return files


def prepare(root):
    files = bundle()
    root.mkdir(parents=True, exist_ok=False)
    for name, contents in files.items():
        (root / name).write_bytes(contents)


def verify(root):
    files = bundle()
    if {p.name for p in root.iterdir()} != set(files):
        raise ValueError("Unexpected or missing bundle file")
    for name, contents in files.items():
        if (root / name).read_bytes() != contents:
            raise ValueError(f"Frozen bundle drift: {name}")
    return json.loads(files["audit.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out)
    print(json.dumps(verify(args.out), indent=2))


if __name__ == "__main__":
    main()
