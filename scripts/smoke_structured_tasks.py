"""Small, priced Super inference smoke on validation data; never trains models."""

import argparse
from dataclasses import asdict
from pathlib import Path

from bench.tasks.structured import sha, write_json
from core.config import Settings
from core.connectors.token_factory import PRICES_USD_PER_M, TokenFactory
from core.eval.data import read_examples
from core.eval.metrics import get_metric
from core.eval.structured import TASK_EVALUATORS, structured_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tasks", type=Path, default=Path("datasets/structured-mixed-v1")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tf = TokenFactory.from_settings(Settings(_env_file=args.env_file))
    tf.client = tf.client.with_options(max_retries=0, timeout=120)
    model = tf.model("super")
    if model not in PRICES_USD_PER_M:
        raise ValueError("Missing model price")
    total = 0.0
    results = {}
    for task in ("json_extraction", "function_calling"):
        path = args.tasks / task / "validation.jsonl"
        examples = read_examples(path)
        # Preselected: three local and three Super-origin prompts, validation only.
        chosen = [examples[i] for i in (0, 1, 2, 96, 97, 98)]
        records = []
        for example in chosen:
            reserve = (
                (sum(len(m["content"].encode()) for m in example.prompt) + 1024)
                * PRICES_USD_PER_M[model][0]
                + 2048 * PRICES_USD_PER_M[model][1]
            ) / 1e6
            if total + reserve > 0.1:
                raise ValueError("Smoke budget USD 0.10 would be exceeded")
            call = tf.chat(
                model,
                example.prompt,
                max_tokens=2048,
                temperature=0,
                response_format={"type": "json_object"},
            )
            total += call.cost_usd
            write_json(args.out / f"{task}-{example.line}-call.json", asdict(call))
            write_json(args.out / "usage.json", {"cost_usd": total, "budget_usd": 0.1})
            records.append(
                {
                    "line": example.line,
                    "answer": call.text,
                    "expected": example.expected,
                    "score": get_metric(TASK_EVALUATORS[task][1])(
                        call.text, example.expected
                    ),
                }
            )
        results[task] = {
            "dataset_sha256": sha(path),
            "split": "validation",
            "model": model,
            "trained": False,
            "records": records,
            "metrics": structured_summary(
                task, [r["answer"] for r in records], [r["expected"] for r in records]
            ),
        }
        write_json(args.out / "results.json", results)
        print(task, results[task]["metrics"], flush=True)


if __name__ == "__main__":
    main()
