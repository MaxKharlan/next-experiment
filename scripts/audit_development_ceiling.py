"""Read-only audit of the stopped development check; never opens a test split."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from bench.tasks.structured import sha
from core.case import EvaluatorSpec
from core.eval.data import read_examples
from core.eval.structured import json_fields_exact, validate_answer
from scripts.fresh_answer_calibration import save


def normalized(text):
    text = re.sub(r"T-\d{6}", "T-ID", text)
    return re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", text)


def audit(old):
    protocol = json.loads((old / "protocol.json").read_text())
    data = {}
    for role in ("train", "validation", "check"):
        item = protocol["source_datasets"][role]
        if sha(Path(item["path"])) != item["sha256"]:
            raise ValueError("Historical data changed")
        data[role] = read_examples(item["path"])
    check = data["check"]
    mutations = Counter()
    for example in check:
        value = validate_answer("json_extraction", example.expected)
        wrong = {
            "ticket_id": "T-999999",
            "product": "scanner" if value["product"] != "scanner" else "router",
            "priority": "low" if value["priority"] != "low" else "high",
            "callback": not value["callback"],
            "due_date": "2030-01-01" if value["due_date"] is None else None,
        }
        for field, replacement in wrong.items():
            answer = json.dumps({**value, field: replacement})
            mutations[field] += json_fields_exact(answer, example.expected) == 0
        mutations["truncated"] += (
            json_fields_exact(example.expected[:30], example.expected) == 0
        )
        mutations["reference_accepted"] += (
            json_fields_exact(example.expected, example.expected) == 1
        )
    rescored = {}
    for path in sorted((old / "scores").glob("*.json")):
        rows = json.loads(path.read_text())["result"]["examples"]
        if len(rows) != len(check):
            raise ValueError("Missing saved predictions")
        for expected, row in zip(check, rows, strict=True):
            if (
                expected.expected != row["expected"]
                or json_fields_exact(row["answer"], row["expected"]) != row["score"]
            ):
                raise ValueError("Saved score/reference mismatch")
        rescored[path.stem] = sum(r["score"] for r in rows) / len(rows)
    groups = {}
    for role, examples in data.items():
        values = [validate_answer("json_extraction", e.expected) for e in examples]
        groups[role] = {
            tuple(v[k] for k in ("product", "priority", "callback")) for v in values
        }
    return {
        "historical_protocol_sha256": sha(old / "protocol.json"),
        "data_sha256": {r: protocol["source_datasets"][r]["sha256"] for r in data},
        "counts": {r: len(v) for r, v in data.items()},
        "metric_mutation_checks": dict(mutations),
        "rescored": rescored,
        "check_slot_groups": len(groups["check"]),
        "check_slot_groups_present_in_train": len(groups["check"] & groups["train"]),
        "normalized_user_prompts": {
            r: len({normalized(e.prompt[-1]["content"]) for e in v})
            for r, v in data.items()
        },
        "scope": "Historical train/validation/check only. No protected final test read. Descriptive audit; cannot restart the stopped protocol.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        if (args.out / "protocol.json").exists():
            raise ValueError("Audit already frozen")
        old = json.loads((args.old / "protocol.json").read_text())
        save(args.out / "offline-audit.json", audit(args.old))
        save(
            args.out / "protocol.json",
            {
                "id": "development-ceiling-audit-20261019",
                "purpose": "One base-model historical-check evaluation, plus metric mutation audit. No training or downstream authorization.",
                "data": old["source_datasets"]["check"],
                "evaluator": old["evaluator"],
                "runtime": old["evaluation_runtime"],
                "model": old["model"],
                "source_sha256": {
                    p: sha(Path(p))
                    for p in (
                        "scripts/audit_development_ceiling.py",
                        "core/eval/generate.py",
                        "core/eval/data.py",
                        "core/eval/structured.py",
                    )
                },
                "cost_usd": 0,
                "stopping": "Exactly one greedy base evaluation. No seed, metric or threshold selection. Historical stop remains final.",
            },
        )
        (args.out / "protocol.sha256").write_text(
            sha(args.out / "protocol.json") + "\n"
        )
    else:
        from core.eval import evaluate

        plan = json.loads((args.out / "protocol.json").read_text())
        if (
            sha(args.out / "protocol.json")
            != (args.out / "protocol.sha256").read_text().strip()
        ):
            raise ValueError("Protocol changed")
        for p, digest in plan["source_sha256"].items():
            if sha(Path(p)) != digest:
                raise ValueError("Source changed")
        if sha(Path(plan["data"]["path"])) != plan["data"]["sha256"]:
            raise ValueError("Data changed")
        if (args.out / "base-score.json").exists():
            raise ValueError("Already evaluated")
        result = evaluate(
            plan["data"]["path"],
            EvaluatorSpec.model_validate(plan["evaluator"]),
            base_model=plan["model"],
            runtime=plan["runtime"],
            device="mps",
        )
        save(args.out / "base-score.json", result.to_dict())
        print(
            json.dumps({"base_exact": result.value, "examples": len(result.examples)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
