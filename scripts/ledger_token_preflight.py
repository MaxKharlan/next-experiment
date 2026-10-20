"""Count a frozen development bundle using an existing local tokenizer only."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from core.probes.level0.lengths import count_tokens
from scripts.prepare_ticket_ledger import digest, encode, verify


def measure(root, tokenizer_path):
    from transformers import AutoTokenizer

    verify(root)
    if not tokenizer_path.is_dir():
        raise ValueError("Supply an existing local tokenizer directory")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    protocol = json.loads((root / "protocol.json").read_text())
    rows = {}
    for path in sorted(root.glob("*.jsonl")):
        messages = [
            json.loads(line)["messages"] for line in path.read_text().splitlines()
        ]
        lengths = [count_tokens(tokenizer, row) for row in messages]
        prompt_lengths = []
        for row in messages:
            ids = tokenizer.apply_chat_template(
                row[:-1],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            prompt_lengths.append(len(ids))
        if max(max(lengths), max(prompt_lengths) + 192) > 8192:
            raise ValueError("Context overflow; truncation forbidden")
        rows[path.name] = {
            "rows": len(messages),
            "rendered_tokens": sum(lengths),
            "max_tokens": max(lengths),
            "max_prompt_plus_generation": max(prompt_lengths) + 192,
        }
    epochs, seeds = protocol["recipe"]["n_epochs"], len(protocol["order_seeds"])
    # Historical scenario only, not an authenticated current quote or approval.
    rate = 0.40
    expected = sum(
        v["rendered_tokens"] * epochs * seeds * rate / 1e6
        for k, v in rows.items()
        if k.endswith("-train.jsonl")
    )
    reservation = sum(
        v["max_tokens"] * v["rows"] * epochs * seeds * rate / 1e6
        for k, v in rows.items()
        if k.endswith("-train.jsonl")
    )
    return {
        "protocol_sha256": digest((root / "protocol.json").read_bytes()),
        "tokenizer_revision": tokenizer_path.name,
        "tokenizer_sha256": {
            p.name: digest(p.read_bytes())
            for p in sorted(tokenizer_path.iterdir())
            if p.name
            in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
        },
        "transformers_version": importlib.metadata.version("transformers"),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "length_counter_sha256": digest(
            Path("core/probes/level0/lengths.py").read_bytes()
        ),
        "files": rows,
        "training_scenario": {
            "slots": 12,
            "historical_usd_per_million": rate,
            "rate_source": "docs/format-content-development-20261019.md (19 October observation)",
            "rendered_token_estimate_usd": expected,
            "max_row_reservation_estimate_usd": reservation,
            "current_price_verified": False,
            "excludes": "Inference, tax, retries, contingency and historical pending holds.",
        },
        "context_check": "passed",
        "admission": "not_measured",
        "paid_authorization_usd": 0,
        "service_spend_usd": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.root, args.tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as file:
        file.write(encode(result))
    print(json.dumps(result["training_scenario"], indent=2))


if __name__ == "__main__":
    main()
