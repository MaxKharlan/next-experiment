"""Record a bounded Super augmentation for deterministic dataset rebuilding."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from bench.tasks.structured import ROLES, sample, write_json
from core.config import Settings
from core.connectors.token_factory import PRICES_USD_PER_M, TokenFactory, extract_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=args.resume)
    settings = Settings(_env_file=args.env_file)
    tf = TokenFactory.from_settings(settings)
    tf.client = tf.client.with_options(max_retries=0, timeout=120)
    model = tf.model("super")
    if model not in PRICES_USD_PER_M:
        raise ValueError("Unknown price; refusing unbudgeted request")
    total = sum(
        json.loads(p.read_text())["cost_usd"] for p in args.out.glob("*-call.json")
    )
    saved = args.out / "augmentation.json"
    batches = json.loads(saved.read_text())["batches"] if saved.exists() else []
    for task in ("json_extraction", "function_calling"):
        for role in ROLES:
            if any(b["task"] == task and b["split"] == role for b in batches):
                continue
            if (args.out / f"{task}-{role}-call.json").exists():
                raise ValueError(
                    "An unaccepted call exists; archive it before retrying"
                )
            rows = [
                sample(task, 900 + i, role, random.Random(100 + i)) for i in range(6)
            ]
            source = [row["messages"][1]["content"] for row in rows]
            request = [
                {
                    "role": "system",
                    "content": "Rewrite each synthetic request in natural English. Preserve every fact, all identifiers, dates, product names, priorities, reasons, locations, and whether a callback is wanted. Do not add facts. Return ONLY a JSON object with one key requests, whose value is an array of six strings in the same order. No explanation or thinking.",
                },
                {"role": "user", "content": json.dumps(source)},
            ]
            # UTF-8 bytes bound input tokens conservatively; output is capped.
            prices = PRICES_USD_PER_M[model]
            reserve = (
                sum(len(m["content"].encode()) for m in request) + 1024
            ) * prices[0] / 1e6 + 4096 * prices[1] / 1e6
            if total + reserve > 0.5:
                raise ValueError("Generation would exceed USD 0.50 reservation")
            call = tf.chat(
                model,
                request,
                max_tokens=4096,
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            total += call.cost_usd
            # Persist usage and original response before parsing or checking content.
            write_json(args.out / f"{task}-{role}-call.json", asdict(call))
            write_json(
                args.out / "usage.json",
                {
                    "cost_usd": total,
                    "budget_usd": 0.5,
                    "basis": "repository configured per-token prices, not provider invoice",
                },
            )
            result = json.loads(extract_json(call.text))
            if isinstance(result, dict):
                result = result.get(
                    "requests", result.get("rewrites", result.get("texts"))
                )
            if (
                not isinstance(result, list)
                or len(result) != len(rows)
                or not all(isinstance(t, str) and 5 < len(t) < 2000 for t in result)
            ):
                raise ValueError("Expected exactly six bounded strings")
            for row, text in zip(rows, result):
                truth = json.loads(row["messages"][-1]["content"])
                identity = truth.get("ticket_id") or truth["arguments"]["order_id"]
                if identity not in text:
                    raise ValueError("Super removed the entity identifier")
                row["messages"][1]["content"] = text
            batches.append(
                {
                    "task": task,
                    "split": role,
                    "rows": rows,
                    "source_prompts": source,
                    "call_file": f"{task}-{role}-call.json",
                }
            )
            write_json(
                args.out / "augmentation.json",
                {
                    "model": model,
                    "prompt_version": "structured-rephrase-v1",
                    "batches": batches,
                },
            )
            print(task, role, f"cost_so_far={total:.6f}", flush=True)


if __name__ == "__main__":
    main()
