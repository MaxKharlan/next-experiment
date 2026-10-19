"""Check every dataset reference against a text-only rule oracle and save metrics."""

import argparse
import json
from pathlib import Path

from bench.tasks.structured import ROLES, sha, validate_bundle, write_json
from bench.tasks.structured_review import parse_request
from core.eval.data import read_examples
from core.eval.structured import structured_summary


def review(root, out):
    out.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for task in ("json_extraction", "function_calling"):
        directory = root / task
        validate_bundle(directory)
        task_rows = []
        for role in ROLES:
            examples = read_examples(directory / f"{role}.jsonl")
            predictions = [
                json.dumps(parse_request(task, e.prompt[-1]["content"]), sort_keys=True)
                for e in examples
            ]
            expected = [e.expected for e in examples]
            scores = structured_summary(task, predictions, expected)
            if scores["exact_match"] != 1.0:
                raise ValueError(f"Reference mismatch: {task}/{role}")
            task_rows.append(
                {
                    "split": role,
                    "examples": len(examples),
                    "metrics": scores,
                    "data_sha256": sha(directory / f"{role}.jsonl"),
                    "predictions": predictions,
                }
            )
        write_json(out / f"{task}.json", task_rows)
        summaries[task] = [
            {"split": r["split"], "examples": r["examples"], "metrics": r["metrics"]}
            for r in task_rows
        ]
    write_json(
        out / "review.json",
        {
            "reviewer": "Codex agent",
            "scope": "All text/reference pairs checked by a bounded rule oracle; all 48 Super paraphrases also read by the agent",
            "human_review": False,
            "learned_model_evaluation": False,
            "summary": summaries,
        },
    )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    review(args.root, args.out)


if __name__ == "__main__":
    main()
