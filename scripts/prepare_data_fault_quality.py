"""Freeze disjoint development data and corruption/repair rules before training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from bench.faults.foreign_template import PRODUCER, repair_declared_wrapper
from bench.faults.semantic_data import (
    contradictory_labels,
    resample_label_distribution,
    template_mismatch,
    validation_overlap,
)
from core.eval.runtime import file_digest
from core.probes.level0.overlap import normalize


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def prepare(root, source):
    if root.exists():
        raise FileExistsError("Use a fresh prospective study directory")
    previous = Path("docs/results/semantic-audit-20261020")
    audit_plan = json.loads((previous / "protocol.json").read_text())
    if file_digest(source) != audit_plan["source_sha256"]:
        raise ValueError("BANKING77 source drift")
    labels = audit_plan["labels"]
    excluded = {
        normalize(row["text"])
        for name in ("reference.json", "audit.json")
        for row in json.loads((previous / name).read_text())
    }
    groups = {label: [] for label in labels}
    for line, row in enumerate(csv.DictReader(source.open()), 2):
        key = normalize(row["text"])
        if row["category"] in groups and key not in excluded:
            groups[row["category"]].append(
                {"text": row["text"], "label": row["category"], "source_line": line}
            )
            excluded.add(key)
    source_splits = {"train": [], "validation": [], "check": []}
    for rows in groups.values():
        rows.sort(
            key=lambda row: hashlib.sha256(
                ("data-fault-quality-6100:" + row["text"]).encode()
            ).hexdigest()
        )
        if len(rows) < 56:
            raise ValueError("Insufficient independent development queries")
        source_splits["train"].extend(rows[:32])
        source_splits["validation"].extend(rows[32:40])
        source_splits["check"].extend(rows[40:56])
    system = (
        "Classify the banking query. Reply with exactly one label, without explanation.\n"
        "Allowed labels:\n" + "\n".join(labels)
    )
    datasets = {}
    for split, rows in source_splits.items():
        random.Random(6100).shuffle(rows)
        datasets[split] = [
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": row["text"]},
                    {"role": "assistant", "content": row["label"]},
                ],
                "source": row["label"],
            }
            for row in rows
        ]
        write_rows(root / f"{split}.jsonl", datasets[split])
    save(root / "source-rows.json", source_splits)
    train = datasets["train"]
    variants = {"healthy": train}
    injections = {}
    for percent in (25, 50):
        for kind, result in (
            ("foreign", template_mismatch(train, percent / 100, 6104)),
            (
                "labels",
                contradictory_labels(train, labels, percent / 100, 6104),
            ),
        ):
            name = f"{kind}-{percent}"
            variants[name], injections[name] = result
    repaired, observed = repair_declared_wrapper(
        variants["foreign-50"], producer=PRODUCER
    )
    assert repaired == train
    injections["foreign-repair-observed"] = observed
    # This exact reconstruction reuses the healthy arm, prospectively.
    variants["foreign-control"] = list(variants["foreign-50"])
    random.Random(6105).shuffle(variants["foreign-control"])
    for name, mass in (("shift-mild", 0.4), ("shift-strong", 0.7)):
        target = {label: (1 - mass) / 7 for label in labels}
        target[labels[0]] = mass
        variants[name] = resample_label_distribution(train, target, 256, 6106)
        injections[name] = target
    variants["shift-repair"] = resample_label_distribution(
        variants["shift-strong"], dict.fromkeys(labels, 1 / 8), 256, 6107
    )
    variants["shift-control"] = list(variants["shift-strong"])
    random.Random(6108).shuffle(variants["shift-control"])
    for percent in (25, 50):
        contaminated, mapping = validation_overlap(
            train, datasets["validation"], percent / 100, 6109
        )
        write_rows(root / f"overlap-validation-{percent}.jsonl", contaminated)
        injections[f"overlap-{percent}"] = mapping
    for name, rows in variants.items():
        write_rows(root / "training" / f"{name}.jsonl", rows)
    save(root / "injections.json", injections)
    embedding = root / "embedding-input"
    save(
        embedding / "reference.json",
        [
            {"id": f"train-{i:03}", "text": row["text"]}
            for i, row in enumerate(source_splits["train"])
        ],
    )
    save(embedding / "audit.json", [])
    save(
        embedding / "protocol.json",
        {
            "id": "data-fault-quality-6100",
            "embedding": audit_plan["embedding"],
            "data_sha256": {
                name: file_digest(embedding / name)
                for name in ("reference.json", "audit.json")
            },
        },
    )
    protocol = {
        "id": "data-fault-quality-6100",
        "source_sha256": file_digest(source),
        "attribution": audit_plan["attribution"],
        "splits": {"train": 256, "validation": 64, "check": 128},
        "exclusion": "All 600 prior semantic-audit queries; normalized text disjoint. Original BANKING77 train only. No final data.",
        "seeds": [6101, 6102, 6103],
        "hp": {
            "n_epochs": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.0002,
            "warmup_ratio": 0,
            "weight_decay": 0,
            "max_grad_norm": 1,
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "context_length": 512,
        },
        "backend": "CUDA float32, isolated runtime; launch only after environment qualification. Freeze runtime and all training source hashes before submission.",
        "generation": {"max_new_tokens": 32, "greedy": True, "thinking": False},
        "quality": "Exact label on same clean 128-query check. Per-seed differences, EOS/limit share and per-class results; descriptive three-seed development comparison.",
        "label_repair": "Reused provider embeddings of 400 external references plus new 256 train vectors. Logistic C=1 max_iter=1000 five-fold shuffled OOF seed 6110; cleanlab prune_by_noise_rate on observed labels. Drop flagged training rows versus equal-count seeded random removal. Original source labels/injected indices forbidden as repair input. Empty/all-row flags recorded as unavailable repair, no threshold retuning.",
        "foreign_repair": "Exact declared role-lines producer reconstruction; healthy training file is identical and reused. Control shuffles the original wrapped rows. No arbitrary quoted-text repair.",
        "overlap": "Training unchanged: reuse healthy adapters. Measure contaminated validation versus clean validation and check; remove observed prompt overlaps versus seeded random equal-count removal from validation. Do not claim a training-quality change.",
        "shift_repair": "Resample observed strong-skew data to declared uniform distribution versus shuffled skew. Fixed examples and epochs; report input/supervised token counts and wall times, not an equal-cost assertion.",
        "stopping": "Measure all frozen arms even when effects are absent. No adaptive data, thresholds, epochs or seeds. No new calibrated fault signatures or 80% power claim from this constructed study.",
        "source_code_sha256": {
            name: file_digest(name)
            for name in (
                "scripts/prepare_data_fault_quality.py",
                "bench/faults/semantic_data.py",
                "bench/faults/foreign_template.py",
            )
        },
        "files_sha256": {
            str(path.relative_to(root)): file_digest(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        },
    }
    save(root / "preparation.json", protocol)
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(args.root, args.source)
    print(json.dumps({"id": plan["id"], "splits": plan["splits"]}))
