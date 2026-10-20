"""Evaluate one frozen reference model's grid on CUDA without retraining."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def choose(scores, jobs):
    if not scores or set(scores) != set(jobs):
        raise ValueError("Every grid point must be scored before selection")
    return min(scores, key=lambda name: (-scores[name], jobs[name]["grid_index"]))


def evaluate(root, output, plan, snapshot, task, role, name, entry):
    import torch

    from core.eval.data import read_examples
    from core.eval.generate import answer, load_model
    from core.eval.metrics import get_metric
    from core.eval.structured import TASK_EVALUATORS

    spec = plan["tasks"][task][role]
    examples = read_examples(root / spec["path"])
    assert digest(root / spec["path"]) == spec["sha256"]
    path = output / "scores" / f"{name}-{role}.json"
    if path.exists():
        raise FileExistsError("Predictions already exist; no automatic repeat")
    adapter = root / entry["adapter"] if entry else None
    model, tokenizer, _ = load_model(adapter, base_model=str(snapshot))
    model.to("cuda").eval()
    metric = get_metric(TASK_EVALUATORS[task][1])
    record = {
        "protocol_sha256": digest(root / "experiment.json"),
        "original_protocol_sha256": plan["original_protocol_sha256"],
        "data_sha256": spec["sha256"],
        "runtime": plan["runtime"],
        "device": "cuda",
        "job_id": entry.get("job_id"),
        "weights_sha256": entry.get("weights_sha256"),
        "rows": [],
        "complete": False,
    }
    started = time.monotonic()
    save(path, record)
    with torch.inference_mode():
        for example in examples:
            record["rows"].append(
                asdict(
                    answer(
                        model,
                        tokenizer,
                        example,
                        "cuda",
                        metric,
                        max_new_tokens=plan["generation"]["max_new_tokens"][task],
                    )
                )
            )
            if len(record["rows"]) % 20 == 0:
                record["elapsed_seconds"] = time.monotonic() - started
                save(path, record)
                print(name, role, len(record["rows"]), "/", len(examples), flush=True)
    rows = record["rows"]
    record.update(
        complete=True,
        elapsed_seconds=time.monotonic() - started,
        exact_match=sum(row["score"] for row in rows) / len(rows),
        stopped_on_eos=sum(row["stopped_on_eos"] for row in rows) / len(rows),
        hit_limit=sum(row["hit_limit"] for row in rows) / len(rows),
    )
    save(path, record)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return record["exact_match"]


def score_pairs(root, output, plan, snapshot, evaluator=evaluate):
    pairs = {}
    for task in plan["tasks"]:
        key = f"{plan['runtime']['model'].split('/')[-1]}-{task}"
        base = {
            role: evaluator(root, output, plan, snapshot, task, role, key + "-base", {})
            for role in ("validation", "check")
        }
        jobs = {
            name: spec for name, spec in plan["jobs"].items() if spec["task"] == task
        }
        scores = {
            name: evaluator(
                root,
                output,
                plan,
                snapshot,
                task,
                "validation",
                name,
                plan["state"][name],
            )
            for name in jobs
        }
        best = choose(scores, jobs)
        choice = {
            "protocol_sha256": digest(root / "experiment.json"),
            "base": base,
            "validation": scores,
            "selected": best,
            "selection_rule": plan["selection"],
        }
        save(output / "selection" / f"{key}.json", choice)
        check = evaluator(
            root, output, plan, snapshot, task, "check", best, plan["state"][best]
        )
        pairs[key] = {**choice, "selected_check_exact_match": check}
        save(output / "results.json", {"status": "partial", "pairs": pairs})
    save(
        output / "results.json",
        {"status": "complete", "pairs": pairs, "scope": plan["scope"]},
    )


def run(root, output):
    import torch
    from huggingface_hub import snapshot_download
    from preflight import run as preflight

    plan = json.loads((root / "experiment.json").read_text())
    if (output / "generation-started.json").exists():
        raise FileExistsError("This evaluation has already started")
    report = preflight(root, output)
    if report["status"] != "passed":
        raise RuntimeError(
            "Dependency preflight failed; all errors are in preflight.json"
        )
    snapshot = Path(
        snapshot_download(
            plan["runtime"]["model"], revision=plan["runtime"]["revision"]
        )
    )
    for name, expected in plan["runtime"]["files_sha256"].items():
        assert digest(snapshot / name) == expected, name
    for name, entry in plan["state"].items():
        assert (
            digest(root / entry["adapter"] / "adapter_model.safetensors")
            == entry["weights_sha256"]
        ), name
        template = root / entry["adapter"] / "chat_template.jinja"
        if template.exists():
            assert digest(template) == plan["runtime"]["template_sha256"], name
    save(
        output / "runtime.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "packages": {
                k: importlib.metadata.version(k) for k in plan["dependency_pins"]
            },
            "precision": "IEEE float32; math SDPA; no TF32",
            "model": plan["runtime"]["model"],
            "revision": plan["runtime"]["revision"],
            "protocol_sha256": digest(root / "experiment.json"),
            "original_protocol_sha256": plan["original_protocol_sha256"],
        },
    )
    save(output / "generation-started.json", {"at_unix": time.time(), "attempt": 1})
    score_pairs(root, output, plan, snapshot)
    save(
        output / "manifest.json",
        {
            p.relative_to(output).as_posix(): digest(p)
            for p in sorted(output.rglob("*.json"))
            if p.name != "manifest.json"
        },
    )


if __name__ == "__main__":
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    run(Path(__file__).parent, Path(os.environ["NEBIUS_OUTPUT_DIR"]) / "evaluation")
