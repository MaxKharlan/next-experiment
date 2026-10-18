"""A1: reproducible BANKING77 chat splits and their file manifest.

    python -m bench.tasks.banking77 --out artifacts/tasks/banking77

Only downloads the public dataset; it does not call models or train jobs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import httpx

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE = "https://github.com/PolyAI-LDN/task-specific-datasets"
DATA_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{REVISION}/banking_data/{{split}}.csv"
)
SYSTEM = (
    "Classify the banking customer query into one of 77 intents. "
    "Answer with the intent label only."
)


def query_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_csv(data: bytes) -> list[tuple[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if reader.fieldnames != ["text", "category"]:
        raise ValueError("expected CSV columns text,category")
    rows = []
    for row in reader:
        text, label = row["text"], row["category"]
        if None in row or not text or not label or not query_key(text):
            raise ValueError("expected a nonempty query and intent label")
        rows.append((text, label))
    if not rows:
        raise ValueError("dataset must not be empty")
    return rows


def split_rows(
    train: list[tuple[str, str]],
    test: list[tuple[str, str]],
    seed: int,
    validation_per_label: int,
    check_per_label: int,
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    """Keep the official test pool held out; deduplicate by normalized query.

    Test queries take precedence over training queries. Conflicting labels fail
    the build rather than silently changing the task's ground truth.
    """
    if validation_per_label < 1 or check_per_label < 1:
        raise ValueError("validation and check counts must be positive")
    seen: dict[str, str] = {}
    pools: dict[str, dict[str, list[tuple[str, str]]]] = {}
    dropped = {"train": 0, "test": 0}
    for name, rows in (("test", test), ("train", train)):
        groups = defaultdict(list)
        for text, label in rows:
            key = query_key(text)
            if key in seen:
                if seen[key] != label:
                    raise ValueError("the same query has conflicting labels")
                dropped[name] += 1
                continue
            seen[key] = label
            groups[label].append((text, label))
        pools[name] = groups
    if not pools["train"] or set(pools["train"]) != set(pools["test"]):
        raise ValueError("train and test must contain the same intent labels")
    rng = random.Random(seed)
    splits: dict[str, list[tuple[str, str]]] = {
        name: [] for name in ("train", "validation", "check", "test")
    }
    for label in sorted(pools["train"]):
        rows = sorted(pools["train"][label])
        n = validation_per_label + check_per_label
        if len(rows) <= n:
            raise ValueError(f"not enough training examples for {label!r}")
        rng.shuffle(rows)
        splits["validation"].extend(rows[:validation_per_label])
        splits["check"].extend(rows[validation_per_label:n])
        splits["train"].extend(rows[n:])
        splits["test"].extend(sorted(pools["test"][label]))
    for rows in splits.values():
        rng.shuffle(rows)
    return splits, dropped


def build(
    train_csv: bytes,
    test_csv: bytes,
    out: Path,
    seed: int = 0,
    validation_per_label: int = 10,
    check_per_label: int = 10,
) -> dict:
    splits, dropped = split_rows(
        read_csv(train_csv),
        read_csv(test_csv),
        seed,
        validation_per_label,
        check_per_label,
    )
    manifest = {
        "task": "banking77",
        "version": 1,
        "source": {
            "repository": SOURCE,
            "revision": REVISION,
            "license": "CC BY 4.0",
            "sha256": {
                "train.csv": hashlib.sha256(train_csv).hexdigest(),
                "test.csv": hashlib.sha256(test_csv).hexdigest(),
            },
        },
        "seed": seed,
        "validation_per_label": validation_per_label,
        "check_per_label": check_per_label,
        "duplicates_removed": dropped,
        "evaluator": {"name": "intent_label", "metric": "exact_match"},
        "datasets": {},
    }
    # A new directory keeps earlier benchmark files and manifests intact.
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in splits.items():
        records = [
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": label},
                ]
            }
            for text, label in rows
        ]
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        raw = data.encode("utf-8")
        path = out / f"{name}.jsonl"
        path.write_bytes(raw)
        manifest["datasets"][name] = {
            "ref": path.name,
            "format": "chat_jsonl",
            "examples": len(rows),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/tasks/banking77"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-per-label", type=int, default=10)
    parser.add_argument("--check-per-label", type=int, default=10)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists; choose a new --out")
    sources = []
    for split in ("train", "test"):
        response = httpx.get(DATA_URL.format(split=split), timeout=60)
        response.raise_for_status()
        sources.append(response.content)
    manifest = build(
        *sources,
        args.out,
        args.seed,
        args.validation_per_label,
        args.check_per_label,
    )
    for name, spec in manifest["datasets"].items():
        print(f"{name}: {spec['examples']} examples")
    print(args.out / "manifest.json")


if __name__ == "__main__":
    main()
