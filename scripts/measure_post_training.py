"""B6: run one small LoRA job on Token Factory and report its time, tokens and files.

Builds a BANKING77 subset in chat format, uploads it, starts the job, polls it, then
prints the duration, trained tokens, checkpoints with metrics and their result_files.
The dollar cost is not in the API: read it in Token Factory Billing after the job.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError

from core.connectors.token_factory import TERMINAL_STATUSES, TokenFactory

# Original BANKING77 release by PolyAI, CC BY 4.0
DATA_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/{split}.csv"
)
SYSTEM = (
    "Classify the banking customer query into one of 77 intents. "
    "Answer with the intent label only."
)


def load_split(split: str) -> list[tuple[str, str]]:
    resp = httpx.get(DATA_URL.format(split=split), timeout=60, follow_redirects=True)
    resp.raise_for_status()
    return [(r["text"], r["category"]) for r in csv.DictReader(io.StringIO(resp.text))]


def to_chat(text: str, label: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
            {"role": "assistant", "content": label},
        ]
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text("".join(lines), encoding="utf-8")


def count_tokens(rows: list[dict[str, Any]], model: str) -> int | None:
    """Tokens in the chat template of the base model; None without transformers."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
        total = 0
        for r in rows:
            text = tok.apply_chat_template(r["messages"], tokenize=False)
            total += len(tok(text, add_special_tokens=False)["input_ids"])
    except (ImportError, OSError):
        # no transformers/jinja2, or the tokenizer repo is gated
        return None
    return total


def describe_files(names: list[str]) -> str:
    if any("adapter_model" in n or "adapter_config" in n for n in names):
        return "LoRA adapter"
    if any(n.endswith(".safetensors") or "pytorch_model" in n for n in names):
        return "merged weights"
    return "unknown, see the file list"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--valid", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="artifacts/b6")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the files and count tokens without starting a job",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train = [to_chat(*row) for row in rng.sample(load_split("train"), args.train)]
    valid = [to_chat(*row) for row in rng.sample(load_split("test"), args.valid)]
    out = Path(args.out)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(valid, out / "valid.jsonl")
    print(f"train {len(train)} examples, valid {len(valid)}, files in {out}/")

    tokens = count_tokens(train, args.model)
    if tokens is None:
        print(
            "token count skipped: run with `uv run --group ml --with jinja2` to count it"
        )
    else:
        print(f"train tokens: {tokens} per epoch, {tokens * args.epochs} in total")
    if args.dry_run:
        return

    tf = TokenFactory.from_settings()
    hyperparameters = {
        "n_epochs": args.epochs,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "lora": True,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "packing": False,
        "context_length": 8192,
    }
    job_id = tf.create_job(
        args.model,
        tf.upload(out / "train.jsonl"),
        validation_file=tf.upload(out / "valid.jsonl"),
        hyperparameters=hyperparameters,
        seed=args.seed,
        suffix="b6-measure",
    )
    (out / "job.json").write_text(
        json.dumps(
            {"job_id": job_id, "hyperparameters": hyperparameters},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Created job: {job_id}", flush=True)

    tf.client = tf.client.with_options(timeout=30, max_retries=0)
    started = time.monotonic()
    deadline = started + 3600
    seen_events: set[str] = set()

    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Poll {job_id} is stopped after a hour. Server task is not canceled."
            )

        try:
            job = tf.job(job_id)
        except (APIConnectionError, APIStatusError) as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code is not None and status_code != 429 and status_code < 500:
                raise
            print(f"Temporary request error: {type(exc).__name__}", flush=True)
            time.sleep(30)
            continue

        done = getattr(job, "trained_steps", None)
        total = getattr(job, "total_steps", None)
        print(
            f"{time.monotonic() - started:.0f}s "
            f"status={job.status} steps={done}/{total} "
            f"tokens={getattr(job, 'trained_tokens', None)}",
            flush=True,
        )

        if job.status in TERMINAL_STATUSES:
            break

        try:
            events = tf.events(job_id)
            for event in sorted(events, key=lambda e: (e.created_at, e.id)):
                if event.id not in seen_events:
                    print(f"[{event.level}] {event.message}", flush=True)
                    seen_events.add(event.id)
        except (APIConnectionError, APIStatusError) as exc:
            print(f"Cannot read the events: {type(exc).__name__}", flush=True)

        time.sleep(30)

    summary: dict[str, Any] = {
        "job_id": job.id,
        "status": job.status,
        "model": args.model,
        "train_examples": len(train),
        "epochs": args.epochs,
        "counted_tokens": tokens * args.epochs if tokens else None,
        "trained_tokens": getattr(job, "trained_tokens", None),
        "trained_steps": getattr(job, "trained_steps", None),
        "server_seconds": job.finished_at - job.created_at if job.finished_at else None,
        "wall_seconds": round(time.monotonic() - started),
        "checkpoints": [],
    }
    if job.status != "succeeded":
        summary["error"] = str(job.error)
        print(f"job ended with {job.status}: {job.error}")
    else:
        for ck in tf.checkpoints(job_id):
            file_ids = getattr(ck, "result_files", None) or []
            files = [tf.client.files.retrieve(fid) for fid in file_ids]
            names = [f.filename for f in files]
            summary["checkpoints"].append(
                {
                    "id": ck.id,
                    "step": ck.step_number,
                    "metrics": ck.metrics.model_dump(mode="json"),
                    "files": {f.filename: f.bytes for f in files},
                    "kind": describe_files(names),
                }
            )
            print(f"checkpoint {ck.id} step {ck.step_number}: {describe_files(names)}")
            print(f"  metrics {ck.metrics}")
            for f in files:
                print(f"  {f.filename}  {f.bytes / 1e6:.1f} MB")

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"server time {summary['server_seconds']} s, tokens {summary['trained_tokens']}"
    )
    print(f"summary saved to {out}/summary.json")
    print("Cost: Token Factory > Billing, find this job; price per 1M = cost / tokens")


if __name__ == "__main__":
    main()
