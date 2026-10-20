"""Fresh three-seed pretrained comparison of the four mode-B trainer switches."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

from core.eval.data import read_examples
from core.eval.generate import answer, free_cache, load_model
from core.eval.metrics import exact_match
from core.eval.runtime import file_digest, freeze_runtime, verify_runtime
from core.interventions.tf_runner import weights_digest
from scripts.fresh_answer_calibration import save

SOURCES = (
    "scripts/pretrained_code_audit.py",
    "jobs/kaggle/train_lora.py",
    "bench/faults/labels.py",
    "core/training/gradients.py",
    "core/training/parameters.py",
    "core/eval/generate.py",
    "core/eval/metrics.py",
)
ARMS = (None, "prompt_loss", "eos_as_pad", "double_shift", "microbatch_mean")


def prepare(root, snapshot):
    from transformers import AutoTokenizer

    from jobs.kaggle.train_lora import encode

    source = Path("docs/results/semantic-audit-20261020/reference.json")
    refs = json.loads(source.read_text())
    labels = list(dict.fromkeys(r["label"] for r in refs))
    groups = {label: [r for r in refs if r["label"] == label] for label in labels}
    splits = {name: [] for name in ("train", "validation", "check")}
    for label in labels:
        for name, start, end in (
            ("train", 0, 12),
            ("validation", 12, 16),
            ("check", 16, 24),
        ):
            for row in groups[label][start:end]:
                splits[name].append(
                    {
                        "id": row["id"],
                        "source": label,
                        "messages": [
                            {
                                "role": "system",
                                "content": "Classify the banking query. Reply with exactly one label, no explanation. Allowed labels: "
                                + ", ".join(labels),
                            },
                            {"role": "user", "content": row["text"]},
                            {"role": "assistant", "content": label},
                        ],
                    }
                )
    root.mkdir(parents=True, exist_ok=False)
    for name, rows in splits.items():
        (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    lengths = [
        len(encode(tokenizer, row["messages"], 8192)["input_ids"])
        for rows in splits.values()
        for row in rows
    ]
    if max(lengths) > 512:
        raise ValueError("Would truncate inputs")
    protocol = {
        "id": "pretrained-code-audit-20261020-v1",
        "source_reference_sha256": file_digest(source),
        "runtime": freeze_runtime(snapshot, "Qwen/Qwen3-0.6B"),
        "datasets_sha256": {
            name: file_digest(root / f"{name}.jsonl") for name in splits
        },
        "source_sha256": {name: file_digest(name) for name in SOURCES},
        "seeds": [4101, 4102, 4103],
        "arms": list(ARMS),
        "hp": {
            "n_epochs": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "context_length": 512,
        },
        "generation": {"thinking": False, "greedy": True, "max_new_tokens": 32},
        "evaluation": "Same 64 development check queries for all 15 adapters plus base; clean validation loss at each epoch. Exact label, EOS/limit breakdown; paired differences across three seeds. Descriptive comparison, no deployment likelihood or 80% recovery guarantee.",
        "stopping": "All four faults measured even if harmful effect absent; retain negative/null results. No adaptive recipe, seeds, threshold or final-test use. No automatic retry of partial training.",
        "splits": {name: len(rows) for name, rows in splits.items()},
        "provider_cost_usd": 0,
        "max_input_tokens": max(lengths),
    }
    save(root / "protocol.json", protocol)
    (root / "protocol.sha256").write_text(file_digest(root / "protocol.json") + "\n")
    print(json.dumps(protocol["splits"]))


def load(root):
    if (
        file_digest(root / "protocol.json")
        != (root / "protocol.sha256").read_text().strip()
    ):
        raise ValueError("Protocol drift")
    p = json.loads((root / "protocol.json").read_text())
    for name, digest in p["source_sha256"].items():
        if file_digest(name) != digest:
            raise ValueError("Source drift: " + name)
    for name, digest in p["datasets_sha256"].items():
        if file_digest(root / f"{name}.jsonl") != digest:
            raise ValueError("Dataset drift")
    verify_runtime(p["runtime"])
    return p


def score(root, output, adapter=None):
    import torch

    p = load(root)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "predictions.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    examples = read_examples(root / "check.jsonl")
    if len(rows) == len(examples):
        return
    model, tok, _ = load_model(adapter, base_model=p["runtime"]["snapshot"])
    model.to("mps").eval()
    for example in examples[len(rows) :]:
        with torch.inference_mode():
            row = asdict(answer(model, tok, example, "mps", exact_match, 32))
        if row["hit_limit"]:
            row["score"] = 0.0
        rows.append(row)
        save(path, rows)
        free_cache("mps")
    save(
        output / "evaluation.json",
        {
            "protocol_sha256": file_digest(root / "protocol.json"),
            "adapter_sha256": weights_digest(adapter) if adapter else None,
            "exact_match": sum(r["score"] for r in rows) / len(rows),
            "eos_share": sum(r["stopped_on_eos"] for r in rows) / len(rows),
            "limit_share": sum(r["hit_limit"] for r in rows) / len(rows),
            "examples": len(rows),
        },
    )
    del model, tok
    gc.collect()
    free_cache("mps")


def run(root, artifacts, max_runs):
    from jobs.kaggle.train_lora import train

    p = load(root)
    artifacts.mkdir(parents=True, exist_ok=True)
    completed = 0
    for seed in p["seeds"]:
        for fault in p["arms"]:
            name = f"{fault or 'healthy'}-s{seed}"
            out = artifacts / name
            if (out / "evaluation.json").exists():
                continue
            if out.exists() and not (out / "summary.json").exists():
                raise RuntimeError(
                    "Partial training requires recorded failure; no automatic retry: "
                    + name
                )
            if not out.exists():
                print("training", name, flush=True)
                out.mkdir()
                save(
                    out / "started.json",
                    {
                        "protocol_sha256": file_digest(root / "protocol.json"),
                        "name": name,
                    },
                )
                train(
                    root / "train.jsonl",
                    root / "validation.jsonl",
                    out,
                    model_id=p["runtime"]["snapshot"],
                    hp=p["hp"],
                    seed=seed,
                    device="mps",
                    code_fault=fault,
                    generate=False,
                    evals_per_epoch=1,
                )
                gc.collect()
                free_cache("mps")
            print("scoring", name, flush=True)
            score(root, out, out / "adapter")
            completed += 1
            if completed >= max_runs:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "base"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--max-runs", type=int, default=1)
    a = parser.parse_args()
    if a.action == "prepare":
        prepare(a.root, a.snapshot)
    elif a.action == "base":
        score(a.root, a.artifacts / "base")
    else:
        run(a.root, a.artifacts, a.max_runs)
