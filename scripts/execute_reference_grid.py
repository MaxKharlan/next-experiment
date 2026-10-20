"""Prepare and execute the clean Post-training reference grid within the TF cap."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from bench.tasks.banking77 import SYSTEM, query_key, read_csv
from core.connectors.storage import LocalStorage
from core.db.spending import SpendingLedger
from core.eval.runtime import PACKAGES, REQUIRED_FILES, file_digest
from core.interventions.tf_runner import download_last_checkpoint, weights_digest
from core.probes.level0.lengths import count_tokens
from scripts.execute_ticket_ledger_v2 import check_accepted
from scripts.fresh_answer_calibration import provider, save

MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
GRID = ((1e-4, 3), (1e-4, 6), (2e-4, 3), (2e-4, 6))
SOURCES = (
    __file__,
    "core/connectors/token_factory.py",
    "core/db/spending.py",
    "core/interventions/tf_runner.py",
    "core/probes/level0/lengths.py",
    "scripts/execute_ticket_ledger_v2.py",
    "scripts/fresh_answer_calibration.py",
    "scripts/score_reference_grid.py",
    "core/eval/generate.py",
    "core/eval/structured.py",
    "core/eval/runtime.py",
)


def settle_estimate(ledger, operation_id, amount):
    entry = next(
        row for row in ledger.snapshot()["operations"] if row["id"] == operation_id
    )
    if entry["status"] == "pending":
        ledger.settle(operation_id, amount)
    elif entry["status"] != "settled" or abs(entry["actual_usd"] - amount) > 1e-12:
        raise ValueError("Conflicting recorded cost")


def reference_runtime(snapshot, model):
    """Bind both single-file and sharded public Qwen snapshots."""
    snapshot = Path(snapshot)
    files = REQUIRED_FILES - {"model.safetensors"}
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards or any(Path(name).name != name for name in shards):
            raise ValueError("Invalid checkpoint shard index")
        files = files | shards | {index.name}
    else:
        files = files | {"model.safetensors"}
    template = json.loads((snapshot / "tokenizer_config.json").read_text())[
        "chat_template"
    ]
    return {
        "model": model,
        "revision": snapshot.name,
        "snapshot": str(snapshot.resolve()),
        "files_sha256": {name: file_digest(snapshot / name) for name in sorted(files)},
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
    }


def verify_reference_runtime(runtime, adapter=None):
    if reference_runtime(runtime["snapshot"], runtime["model"]) != runtime:
        raise ValueError("Evaluation runtime changed")
    if adapter:
        template = Path(adapter) / "chat_template.jinja"
        if template.exists() and file_digest(template) != runtime["template_sha256"]:
            raise ValueError("Adapter template differs from frozen template")


def prepare(root):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    root.mkdir(parents=True, exist_ok=False)
    costs = Path("docs/results/matrix-tokenizer-costs-20261020")
    tokenizers = json.loads((costs / "tokenizers.json").read_text())["models"]
    datasets = root / "datasets"
    datasets.mkdir()
    for task in ("json_extraction", "function_calling"):
        target = datasets / task
        target.mkdir()
        for role in ("train", "validation", "check"):
            shutil.copyfile(
                Path("datasets/structured-mixed-v1") / task / f"{role}.jsonl",
                target / f"{role}.jsonl",
            )
    csv = Path("artifacts/remaining-queue-20261020/banking77-train.csv")
    groups = defaultdict(list)
    seen = {}
    for query, label in read_csv(csv.read_bytes()):
        key = query_key(query)
        if key in seen:
            if seen[key] != label:
                raise ValueError("Conflicting source labels")
            continue
        seen[key] = label
        groups[label].append(query)
    rng = random.Random(7101)
    splits = {role: [] for role in ("train", "validation", "check")}
    for i, label in enumerate(sorted(groups)):
        queries = sorted(groups[label])
        rng.shuffle(queries)
        count = 6 if i < 5 else 5
        for role, rows in (
            ("train", queries[:count]),
            ("validation", queries[count : count + 2]),
            ("check", queries[count + 2 : count + 4]),
        ):
            for query in rows:
                splits[role].append(
                    {
                        "source": label,
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": label},
                        ],
                    }
                )
    bank = datasets / "banking77"
    bank.mkdir()
    for role, rows in splits.items():
        rng.shuffle(rows)
        (bank / f"{role}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    assert len(splits["train"]) == 390
    assert len(splits["validation"]) == len(splits["check"]) == 154
    tasks = {}
    for task in ("banking77", "json_extraction", "function_calling"):
        tasks[task] = {
            role: {
                "path": str((datasets / task / f"{role}.jsonl").resolve()),
                "sha256": file_digest(datasets / task / f"{role}.jsonl"),
            }
            for role in ("train", "validation", "check")
        }
    jobs = {}
    runtimes = {}
    for model in MODELS:
        snapshot = snapshot_download(
            model,
            revision=tokenizers[model]["revision"],
            allow_patterns=sorted(
                REQUIRED_FILES | {"*.safetensors", "model.safetensors.index.json"}
            ),
            local_files_only=True,
        )
        runtimes[model] = reference_runtime(snapshot, model)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        for task, files in tasks.items():
            counts = {}
            for role, spec in files.items():
                rows = [
                    json.loads(line)
                    for line in Path(spec["path"]).read_text().splitlines()
                ]
                lengths = [count_tokens(tokenizer, row["messages"]) for row in rows]
                assert max(lengths) <= 8192
                counts[role] = {
                    "tokens": sum(lengths),
                    "max": max(lengths),
                    "rows": len(rows),
                }
            for index, (lr, epochs) in enumerate(GRID):
                hp = {
                    "n_epochs": epochs,
                    "batch_size": 8,
                    "learning_rate": lr,
                    "lora": True,
                    "lora_r": 16,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "packing": False,
                    "context_length": 8192,
                    "warmup_ratio": 0.0,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                }
                name = f"{model.split('/')[-1]}-{task}-g{index}"
                jobs[name] = {
                    "model": model,
                    "task": task,
                    "grid_index": index,
                    "seed": 7101,
                    "hyperparameters": hp,
                    "counts": counts,
                    "reservation_usd": counts["train"]["max"]
                    * counts["train"]["rows"]
                    * epochs
                    * 0.4
                    / 1e6
                    * 1.2,
                }
    plan = {
        "id": "clean-reference-grid-20261020",
        "tasks": tasks,
        "jobs": jobs,
        "runtimes": runtimes,
        "tokenizer_provenance": {name: tokenizers[name] for name in MODELS},
        "price_observation_sha256": file_digest(costs / "prices.json"),
        "source_sha256": {str(p): file_digest(p) for p in SOURCES},
        "source_banking77_sha256": file_digest(csv),
        "source_banking77": "PolyAI task-specific-datasets; Casanueva et al. 2020; CC BY 4.0. Official training pool only.",
        "selection": "Evaluate all four final recipe checkpoints on validation exact match. Select highest score, ties by grid index. Evaluate only the selected checkpoint once on check. Record base quality separately before selection. No final test.",
        "amendment": "B5 clean-reference selection can proceed independently of answer-cut admission. This runs only the existing four-point healthy grid; no damaged/fixed/control or confirmation series is admitted by this protocol.",
        "scope": "Development references for existing structured integration tasks and a new 77-intent subset. Existing structured splits have already been inspected. No independent final-test or broad recovery claim.",
        "generation": {
            "greedy": True,
            "thinking": False,
            "max_new_tokens": {
                "banking77": 32,
                "json_extraction": 192,
                "function_calling": 192,
            },
        },
        "concurrency": 3,
        "stopping": "One submission per grid point. Stop new submissions after any failed, cancelled, mismatched or ambiguous attempt. Keep completed jobs and all reservations. Do not automatically repeat training.",
        "llama": "User selected Token Factory. Existing provider tokenizer is accessible, but the LoRA export lacks base weights and no 1B inference endpoint is listed. Keep its new quality grid unsubmitted until an evaluation route is available.",
        "ledger": "artifacts/remaining-queue-20261020/spending.sqlite3",
        "cap_usd": 25,
        "usd_per_million": 0.4,
        "allowance": 1.2,
    }
    ledger = SpendingLedger(plan["ledger"], 25)
    reservation = sum(job["reservation_usd"] for job in jobs.values())
    if reservation > ledger.snapshot()["available_usd"]:
        raise ValueError("Clean reference grid exceeds remaining budget")
    save(root / "protocol.json", plan)
    (root / "protocol.sha256").write_text(file_digest(root / "protocol.json") + "\n")
    print(json.dumps({"jobs": len(jobs), "reservation_usd": reservation}), flush=True)


def execute(root, env):
    digest = file_digest(root / "protocol.json")
    if digest != (root / "protocol.sha256").read_text().strip():
        raise ValueError("Protocol changed")
    plan = json.loads((root / "protocol.json").read_text())
    for name, expected in plan["source_sha256"].items():
        if file_digest(name) != expected:
            raise ValueError("Executor source changed")
    for files in plan["tasks"].values():
        for spec in files.values():
            if file_digest(spec["path"]) != spec["sha256"]:
                raise ValueError("Dataset changed")
    state_path = root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    ledger = SpendingLedger(plan["ledger"], plan["cap_usd"])
    tf = provider(env)
    deadline = time.monotonic() + 6 * 3600
    while True:
        active = 0
        for name, entry in state.items():
            if not entry.get("job_id"):
                raise ValueError("Ambiguous submission; no retry")
            if entry.get("settled"):
                continue
            job = tf.job(entry["job_id"])
            spec = plan["jobs"][name]
            check_accepted(
                job,
                spec["model"],
                entry["training_file"],
                {
                    "hyperparameters": spec["hyperparameters"],
                    "seed": spec["seed"],
                    "validation_file": entry["validation_file"],
                },
            )
            save(root / "provider" / f"{name}.json", job.model_dump())
            entry["status"] = job.status
            if job.status in {"failed", "cancelled"}:
                save(state_path, state)
                raise ValueError("Reference job failed; stop further submissions")
            if job.status == "succeeded":
                if not entry.get("adapter"):
                    adapter = download_last_checkpoint(
                        tf, LocalStorage(root), job.id, "checkpoints"
                    )
                    entry.update(
                        adapter=str(adapter.resolve()),
                        weights_sha256=weights_digest(adapter),
                    )
                if type(job.trained_tokens) is not int or job.trained_tokens < 0:
                    raise ValueError("Missing provider token count")
                estimate = job.trained_tokens * plan["usd_per_million"] / 1e6
                settle_estimate(
                    ledger, entry["operation_id"], estimate * plan["allowance"]
                )
                entry.update(
                    settled=True,
                    trained_tokens=job.trained_tokens,
                    token_cost_estimate_usd=estimate,
                )
                print(name, "succeeded", flush=True)
            else:
                active += 1
            save(state_path, state)
        for name, spec in plan["jobs"].items():
            if name in state or active >= plan["concurrency"]:
                continue
            files = plan["tasks"][spec["task"]]
            operation = f"{plan['id']}:{name}"
            ledger.reserve(operation, "clean_reference_grid", spec["reservation_usd"])
            entry = state[name] = {
                "operation_id": operation,
                "status": "submitting",
                "protocol_sha256": digest,
            }
            save(state_path, state)
            entry["training_file"] = tf.upload(Path(files["train"]["path"]))
            entry["validation_file"] = tf.upload(Path(files["validation"]["path"]))
            save(state_path, state)
            kwargs = {key: spec[key] for key in ("hyperparameters", "seed")}
            kwargs.update(
                validation_file=entry["validation_file"],
                suffix="ne-reference-" + str(spec["grid_index"]),
            )
            entry["job_id"] = tf.create_job(
                spec["model"], entry["training_file"], **kwargs
            )
            save(state_path, state)
            accepted = tf.job(entry["job_id"])
            save(root / "accepted" / f"{name}.json", accepted.model_dump())
            try:
                check_accepted(accepted, spec["model"], entry["training_file"], kwargs)
            except ValueError:
                if accepted.status not in {"succeeded", "failed", "cancelled"}:
                    tf.client.fine_tuning.jobs.cancel(accepted.id)
                raise
            entry["status"] = accepted.status
            save(state_path, state)
            active += 1
            print(name, entry["job_id"], flush=True)
        if len(state) == len(plan["jobs"]) and all(
            e.get("settled") for e in state.values()
        ):
            print("REFERENCE_TRAINING_COMPLETE", flush=True)
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                "Reference grid remains unfinished; preserve IDs and holds"
            )
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    with (args.root.parent / "reference-grid.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "prepare":
            prepare(args.root)
        else:
            execute(args.root, args.env_file)
