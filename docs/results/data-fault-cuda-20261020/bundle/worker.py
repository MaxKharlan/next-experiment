"""Run a frozen development training matrix on CUDA and export every outcome."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from core.eval.data import read_examples
from core.eval.generate import answer, load_model
from core.eval.metrics import exact_match
from core.probes.level1.logits import aggregate, inspect_example
from jobs.kaggle.train_lora import encode, train


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def evaluate(snapshot, adapter, data, output, logit_data=None):
    model, tokenizer, _ = load_model(adapter, base_model=str(snapshot))
    model.to("cuda").eval()
    rows = []
    for example in read_examples(data):
        with torch.inference_mode():
            row = asdict(answer(model, tokenizer, example, "cuda", exact_match, 32))
        if row["hit_limit"]:
            row["score"] = 0.0
        rows.append(row)
    save(output / "predictions.json", rows)
    groups = sorted({row["source"] for row in rows})
    metrics = {
        "examples": len(rows),
        "exact_match": sum(row["score"] for row in rows) / len(rows),
        "eos_share": sum(row["stopped_on_eos"] for row in rows) / len(rows),
        "limit_share": sum(row["hit_limit"] for row in rows) / len(rows),
        "by_source": {
            group: {
                "examples": sum(row["source"] == group for row in rows),
                "exact_match": sum(
                    row["score"] for row in rows if row["source"] == group
                )
                / sum(row["source"] == group for row in rows),
            }
            for group in groups
        },
    }
    if logit_data:
        diagnostics = [
            inspect_example(model, tokenizer, example, "cuda", context_length=512)
            for example in read_examples(logit_data)
        ]
        save(output / "logit-rows.json", diagnostics)
        metrics["logit_summary"] = aggregate(diagnostics)
    save(output / "evaluation.json", metrics)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main():
    root = Path(__file__).parent
    plan = json.loads((root / "experiment.json").read_text())
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    for relative, expected in plan["bundle_files_sha256"].items():
        assert sha(root / relative) == expected, relative
    packages = {
        name: importlib.metadata.version(name) for name in plan["runtime"]["packages"]
    }
    for name, expected in plan["runtime"]["packages"].items():
        assert packages[name].split("+")[0] == expected, (name, packages[name])
    snapshot = Path(
        snapshot_download(
            plan["runtime"]["model"], revision=plan["runtime"]["revision"]
        )
    )
    for name, expected in plan["runtime"]["files_sha256"].items():
        assert sha(snapshot / name) == expected, name
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    for path in sorted((root / "data").rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            messages = json.loads(line)["messages"]
            tokens = tokenizer.apply_chat_template(messages, tokenize=True)
            assert len(tokens) <= plan["hp"]["context_length"], "Truncation forbidden"
            encode(tokenizer, messages, plan["hp"]["context_length"])
    del tokenizer
    destination = Path(os.environ["NEBIUS_OUTPUT_DIR"])
    work = Path("/tmp/next-experiment-training")
    work.mkdir(exist_ok=False)
    runtime = {
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "protocol_sha256": sha(root / "experiment.json"),
        "model_revision": plan["runtime"]["revision"],
        "dtype": "torch.float32",
    }
    save(destination / "runtime.json", runtime)
    result = {"id": plan["id"], "status": "running", "runtime": runtime, "cells": {}}
    started = time.monotonic()
    if plan.get("evaluate_base"):
        result["base"] = evaluate(
            snapshot, None, root / "data/check.jsonl", work / "base"
        )
        shutil.copytree(work / "base", destination / "base", dirs_exist_ok=True)
        save(destination / "result.json", result)
    for seed in plan["seeds"]:
        for arm in plan["arms"]:
            name = f"{arm['name']}-s{seed}"
            output = work / name
            output.mkdir()
            try:
                print("training", name, flush=True)
                train(
                    root / arm["train"],
                    root / "data/validation.jsonl",
                    output,
                    model_id=str(snapshot),
                    hp=plan["hp"],
                    seed=seed,
                    device="cuda",
                    code_fault=arm.get("code_fault"),
                    generate=False,
                    evals_per_epoch=1,
                )
                gc.collect()
                torch.cuda.empty_cache()
                metrics = evaluate(
                    snapshot,
                    output / "adapter",
                    root / "data/check.jsonl",
                    output,
                    root / "data/logit-validation.jsonl",
                )
                for relative in arm.get("extra_evaluations", []):
                    evaluate(
                        snapshot,
                        output / "adapter",
                        root / relative,
                        output / "extra" / Path(relative).stem,
                    )
                metrics["weights_sha256"] = sha(
                    output / "adapter/adapter_model.safetensors"
                )
                metrics["summary_sha256"] = sha(output / "summary.json")
                result["cells"][name] = metrics
                save(
                    output / "manifest.json",
                    {
                        str(path.relative_to(output)): sha(path)
                        for path in sorted(output.rglob("*"))
                        if path.is_file()
                    },
                )
                shutil.copytree(output, destination / name, dirs_exist_ok=True)
                result["wall_seconds"] = time.monotonic() - started
                save(destination / "result.json", result)
                print("completed", name, json.dumps(metrics), flush=True)
            except Exception as error:
                result.update(status="failed", failed_cell=name, error=str(error))
                save(destination / "result.json", result)
                shutil.copytree(output, destination / name, dirs_exist_ok=True)
                raise
    result.update(status="complete", wall_seconds=time.monotonic() - started)
    save(destination / "result.json", result)
    print("MATRIX_COMPLETE=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
