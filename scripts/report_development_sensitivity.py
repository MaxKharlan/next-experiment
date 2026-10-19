"""Descriptive score summaries, including a labeled post-hoc format diagnostic."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from bench.tasks.structured import sha
from core.eval.structured import json_fields_exact, json_valid, validate_answer
from scripts.fresh_answer_calibration import save


def summarize(path):
    artifact = json.loads(path.read_text())
    result = artifact.get("result", artifact)
    groups = defaultdict(list)
    for row in result["examples"]:
        wrapped = re.fullmatch(r"```json\s*\n(.*?)\n```\s*", row["answer"], re.DOTALL)
        raw = wrapped[1] if wrapped else row["answer"]
        schema_valid = True
        try:
            validate_answer("json_extraction", row["answer"])
        except (ValueError, TypeError):
            schema_valid = False
        observed = {
            "strict_exact": json_fields_exact(row["answer"], row["expected"]),
            "json_valid": json_valid(row["answer"], row["expected"]),
            "schema_valid": schema_valid,
            "fenced": bool(wrapped),
            "posthoc_unwrapped_content_exact": json_fields_exact(raw, row["expected"]),
            "hit_limit": row["hit_limit"],
        }
        if observed["strict_exact"] != row["score"]:
            raise ValueError("Recorded score mismatch")
        groups["all"].append(observed)
        if row.get("source"):
            groups[row["source"]].append(observed)
    means = {
        group: {
            "n": len(rows),
            **{k: sum(r[k] for r in rows) / len(rows) for k in rows[0]},
        }
        for group, rows in groups.items()
    }
    if means["all"]["strict_exact"] != result["value"]:
        raise ValueError("Recorded mean mismatch")
    return {"score_sha256": sha(path), "groups": means}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    scores = {
        "historical-base": summarize(args.audit / "base-score.json"),
        "candidate-base": summarize(args.pilot / "scores/base.json"),
    }
    scores.update(
        {p.stem: summarize(p) for p in sorted((args.screen / "scores").glob("*.json"))}
    )
    save(
        args.out,
        {
            "scores": scores,
            "diagnostic_boundary": "Markdown unwrapping is post-hoc descriptive analysis only. All preregistered gates and reported primary outcomes use strict json_fields_exact, unchanged.",
            "pilot_decision": json.loads((args.pilot / "decision.json").read_text())
            if (args.pilot / "decision.json").exists()
            else None,
            "screen_decision": json.loads((args.screen / "decision.json").read_text()),
        },
    )


if __name__ == "__main__":
    main()
