"""Development-only specificity controls for answer-cut-v2; no model or provider."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from bench.tasks.structured import sha, write_json
from core.probes.level0.answers import AnswerCut, file_answers

RATES = (1, 5, 10, 25, 50)
KINDS = (
    "clean_labels",
    "clean_prefix_labels",
    "truncated_prefix_labels",
    "clean_json",
    "truncated_json",
    "internal_json_syntax",
    "valid_wrong_json",
)


def fixture(kind, count):
    rows, affected = [], []
    for i in range(100):
        selected = i < count
        if "labels" in kind:
            truth = "pay_later" if i < 60 else "refund"
            if kind == "clean_prefix_labels" and selected:
                truth = "pay"
            observed = (
                "pay" if kind == "truncated_prefix_labels" and selected else truth
            )
            prompt = f"Return the exact category named in this request: {truth}. Request {i}."
        else:
            truth = json.dumps({"id": i, "value": "complete"}, sort_keys=True)
            observed = truth
            if selected and kind == "truncated_json":
                observed = truth[:-4]
            elif selected and kind == "internal_json_syntax":
                observed = truth.replace(",", "", 1)
            elif selected and kind == "valid_wrong_json":
                observed = json.dumps({"id": i, "value": "wrong"}, sort_keys=True)
            prompt = f'Return JSON with id {i} and value "complete".'
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": observed},
                ]
            }
        )
        if observed != truth:
            affected.append(i + 1)
    # Same row order for paired clean-prefix and damaged-prefix cases.
    order = list(range(100))
    random.Random(20261019 + count).shuffle(order)
    return [rows[i] for i in order], [
        j + 1 for j, i in enumerate(order) if i + 1 in affected
    ]


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    cases = {}
    for kind in KINDS:
        for count in RATES:
            name = f"{kind}-{count:02d}"
            rows, damaged = fixture(kind, count)
            path = root / f"{name}.jsonl"
            path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
            cases[name] = {
                "kind": kind,
                "count": count,
                "path": path.name,
                "sha256": sha(path),
                "damaged_lines": damaged,
                "truncation_cause": kind
                in ("truncated_prefix_labels", "truncated_json"),
            }
    write_json(
        root / "protocol.json",
        {
            "id": "answer-cut-specificity-controls-20261019",
            "phase": "synthetic_development_controls",
            "cases": cases,
            "sources": {
                p: sha(Path(p))
                for p in (
                    "bench/answer_cut_controls.py",
                    "core/probes/level0/answers.py",
                    "core/probes/signatures.yaml",
                )
            },
            "analysis": "Run the unmodified probe once per case; report case outcomes, clean-row false flags and truncation-mechanism false alarms separately. Counts are constructed coverage, not population specificity or independent training seeds. No fitting or deployment.",
            "stopping": "Exactly these 35 deterministic fixtures. No adaptive controls, model calls or changes to live likelihoods.",
            "cost_usd": 0,
        },
    )
    (root / "protocol.sha256").write_text(sha(root / "protocol.json") + "\n")


def run(root):
    digest = (root / "protocol.sha256").read_text().strip()
    if sha(root / "protocol.json") != digest:
        raise ValueError("Protocol changed")
    if (root / "results.json").exists():
        raise ValueError("Controls already measured")
    plan = json.loads((root / "protocol.json").read_text())
    for p, expected in plan["sources"].items():
        if sha(Path(p)) != expected:
            raise ValueError("Source changed")
    results = {}
    for name, case in plan["cases"].items():
        path = root / case["path"]
        if sha(path) != case["sha256"]:
            raise ValueError("Fixture changed")
        raw = file_answers(path)
        flagged, damaged = set(raw.cut_lines), set(case["damaged_lines"])
        results[name] = {
            "kind": case["kind"],
            "count": case["count"],
            "outcome": "cut" if raw.cut_share >= AnswerCut.CUT else "ok",
            "correctly_flagged_damaged_rows": len(flagged & damaged),
            "missed_damaged_rows": len(damaged - flagged),
            "false_flagged_clean_rows": len(flagged - damaged),
            "truncation_cause": case["truncation_cause"],
            "raw": asdict(raw),
        }
    pairs = []
    for rate in RATES:
        a, b = (
            results[f"{kind}-{rate:02d}"]
            for kind in ("clean_prefix_labels", "truncated_prefix_labels")
        )
        pairs.append(
            {
                "prefix_count": rate,
                "same_probe_result": a["raw"] == b["raw"],
                "clean_outcome": a["outcome"],
                "damaged_outcome": b["outcome"],
            }
        )
    write_json(
        root / "results.json",
        {
            "protocol_sha256": digest,
            "results": results,
            "paired_controls": pairs,
            "deployment": "Unchanged T=1 and heuristic signature table; controls reject treating cut as unique mechanism evidence. No model-quality or population-generalization claim.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else run)(args.root)


if __name__ == "__main__":
    main()
