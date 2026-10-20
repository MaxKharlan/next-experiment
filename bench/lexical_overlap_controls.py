"""Freeze and measure authored lexical controls without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from core.eval.data import Example
from core.probes.level0.overlap import (
    GRAM_SIZE,
    MIN_CHARACTERS,
    THRESHOLD,
    VERSION,
    find_candidates,
)

CONTROL = Path("bench/controls/lexical-overlap-v1.json")
SOURCES = (
    "core/probes/level0/overlap.py",
    "core/eval/data.py",
    "bench/lexical_overlap_controls.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serialize(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def protocol():
    return {
        "id": "lexical-overlap-controls-v1",
        "method": VERSION,
        "representation": "NFKC/casefold/whitespace normalization, message-indexed character sets",
        "gram_size": GRAM_SIZE,
        "threshold": THRESHOLD,
        "min_characters": MIN_CHARACTERS,
        "fuzzy_context_guard": "Same role layout and normalized system messages",
        "reference_answers": "Excluded from candidate detection",
        "source_sha256": {name: digest(name) for name in SOURCES},
        "controls_sha256": digest(CONTROL),
        "decision": "Publish all confusion counts at the fixed threshold; no tuning or deployment gate",
        "scope": "Constructed stress controls with author labels, not independent semantic accuracy",
        "provider_calls": 0,
    }


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    (root / "controls.json").write_bytes(CONTROL.read_bytes())
    (root / "protocol.json").write_text(serialize(protocol()))


def measure(controls):
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    rows = []
    pairs = controls["pairs"]
    if not pairs or len({pair["id"] for pair in pairs}) != len(pairs):
        raise ValueError("Empty or duplicate control IDs")
    for pair in pairs:
        truth = pair["same_request"]
        if not isinstance(truth, bool):
            raise TypeError("Control truth must be boolean")
        report = find_candidates(
            [Example(1, pair["train_prompt"], "reference-a")],
            [Example(1, pair["validation_prompt"], "reference-b")],
        )
        detected = bool(report["candidate_pairs"])
        outcome = ("t" if detected == truth else "f") + ("p" if detected else "n")
        counts[outcome] += 1
        rows.append(
            {
                "id": pair["id"],
                "category": pair["category"],
                "same_request": truth,
                "candidate_detected": detected,
                "confusion": outcome,
                "candidate_pairs": report["candidate_pairs"],
            }
        )
    return {
        "protocol_id": "lexical-overlap-controls-v1",
        "scope": "Authored controls only; no independent semantic or quality claim",
        "confusion": counts,
        "pairs": rows,
        "provider_calls": 0,
        "service_spend_usd": 0,
    }


def check_protocol(root):
    expected = json.loads((root / "protocol.json").read_text())
    if (
        expected != protocol()
        or digest(root / "controls.json") != expected["controls_sha256"]
    ):
        raise ValueError("Frozen source, controls or protocol changed")


def run(root):
    check_protocol(root)
    report = measure(json.loads((root / "controls.json").read_text()))
    with (root / "report.json").open("x") as file:
        file.write(serialize(report))
    manifest = {
        name: digest(root / name)
        for name in ("protocol.json", "controls.json", "report.json")
    }
    with (root / "manifest.json").open("x") as file:
        file.write(serialize(manifest))
    return report


def verify(root):
    check_protocol(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if set(manifest) != {"protocol.json", "controls.json", "report.json"}:
        raise ValueError("Manifest coverage changed")
    for name, sha in manifest.items():
        if digest(root / name) != sha:
            raise ValueError(f"Evidence changed: {name}")
    report = measure(json.loads((root / "controls.json").read_text()))
    if (root / "report.json").read_text() != serialize(report):
        raise ValueError("Report differs from controls")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out)
    else:
        report = (run if args.action == "run" else verify)(args.out)
        print(json.dumps(report["confusion"]))


if __name__ == "__main__":
    main()
