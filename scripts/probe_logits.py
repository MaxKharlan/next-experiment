"""Run a local A29 teacher-forced audit on explicitly supplied development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.eval.data import read_examples
from core.eval.generate import free_cache, load_model, pick_device
from core.eval.runtime import file_digest, freeze_runtime
from core.interventions.tf_runner import weights_digest
from core.probes.level1.logits import aggregate, inspect_example
from scripts.fresh_answer_calibration import save


def run(data, snapshot, output, *, adapter=None, context_length=8192, device="auto"):
    if output.exists():
        raise FileExistsError(
            "Use a fresh output directory; completed audits are immutable"
        )
    runtime = freeze_runtime(snapshot, "Qwen/Qwen3-0.6B")
    examples = read_examples(data)
    digest = file_digest(data)
    adapter_digest = weights_digest(adapter) if adapter else None
    output.mkdir(parents=True)
    protocol = {
        "id": "teacher-forced-logit-audit-v1",
        "data_sha256": digest,
        "runtime": runtime,
        "adapter_sha256": adapter_digest,
        "context_length": context_length,
        "source_sha256": {
            name: file_digest(name)
            for name in (
                "core/probes/level1/logits.py",
                "scripts/probe_logits.py",
                "core/eval/data.py",
                "core/eval/generate.py",
            )
        },
        "scope": "Explicit development data only. No fitted likelihood, generated-EOS rate or causal confirmation.",
    }
    save(output / "protocol.json", protocol)
    model, tokenizer, _ = load_model(adapter, base_model=str(snapshot))
    chosen_device = pick_device(device)
    model.to(chosen_device).eval()
    rows = []
    for example in examples:
        rows.append(
            inspect_example(
                model, tokenizer, example, chosen_device, context_length=context_length
            )
        )
        save(output / "rows.json", rows)
        free_cache(chosen_device)
    if file_digest(data) != digest or (
        adapter and weights_digest(adapter) != adapter_digest
    ):
        raise ValueError("Input changed during measurement")
    result = {
        "protocol_sha256": file_digest(output / "protocol.json"),
        "device": chosen_device,
        "summary": aggregate(rows),
        "provider_cost_usd": 0,
    }
    save(output / "result.json", result)
    save(
        output / "manifest.json",
        {
            name: file_digest(output / name)
            for name in ("protocol.json", "rows.json", "result.json")
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.data,
                args.snapshot,
                args.out,
                adapter=args.adapter,
                context_length=args.context_length,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
