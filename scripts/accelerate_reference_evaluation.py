"""Package and submit two bounded evaluations using the owner's current Cloud balance."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from core.connectors.compute import JobSpec, ServerlessJobs
from core.db.spending import SpendingLedger
from core.eval.runtime import file_digest
from scripts.execute_reference_grid import verify_reference_runtime
from scripts.stage_cloud_bundle import stage

SOURCE = Path("docs/results/clean-reference-grid-20261020")
DESTINATION = Path("artifacts/remaining-queue-20261020/qwen-acceleration")
CLI = "/Users/harlanmax/.nebius/bin/nebius"
MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def ready(plan, state, model):
    names = {name for name, job in plan["jobs"].items() if job["model"] == model}
    return len(names) == 12 and all(
        name in state
        and state[name].get("settled")
        and state[name]["status"] == "succeeded"
        for name in names
    )


def package(model):
    original_digest = file_digest(SOURCE / "protocol.json")
    assert original_digest == (SOURCE / "protocol.sha256").read_text().strip()
    original = json.loads((SOURCE / "protocol.json").read_text())
    for name, expected in original["source_sha256"].items():
        assert file_digest(name) == expected, name
    assert not list((SOURCE / "scores").glob("*.json"))
    state = json.loads((SOURCE / "state.json").read_text())
    assert ready(original, state, model)
    root = DESTINATION / model.split("/")[-1]
    bundle = root / "bundle"
    bundle.mkdir(parents=True, exist_ok=False)
    runtime = original["runtimes"][model]
    verify_reference_runtime(runtime)
    jobs = {n: j for n, j in original["jobs"].items() if j["model"] == model}
    bound_state = {}
    for name in jobs:
        entry = state[name]
        source = Path(entry["adapter"])
        assert (
            file_digest(source / "adapter_model.safetensors") == entry["weights_sha256"]
        )
        verify_reference_runtime(runtime, source)
        target = bundle / "adapters" / name
        target.mkdir(parents=True)
        for filename in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "chat_template.jinja",
        ):
            if (source / filename).exists():
                shutil.copyfile(source / filename, target / filename)
        bound_state[name] = {**entry, "adapter": target.relative_to(bundle).as_posix()}
    tasks = {}
    for task, specs in original["tasks"].items():
        tasks[task] = {}
        for role in ("validation", "check"):
            spec = specs[role]
            assert file_digest(spec["path"]) == spec["sha256"]
            target = bundle / "data" / task / f"{role}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(spec["path"], target)
            tasks[task][role] = {**spec, "path": target.relative_to(bundle).as_posix()}
    for name in ("data", "generate", "metrics", "records", "structured"):
        target = bundle / "core/eval" / f"{name}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path("core/eval") / f"{name}.py", target)
    shutil.copyfile("jobs/cloud/reference_evaluation.py", bundle / "worker.py")
    shutil.copyfile("jobs/kaggle/evaluation_preflight.py", bundle / "preflight.py")
    pins = json.loads(
        Path("docs/results/kaggle-evaluation-v3-20261020/protocol.json").read_text()
    )["dependency_pins"]
    assert all(pins[k] == v for k, v in runtime["packages"].items())
    (bundle / "dependencies.txt").write_text(
        "".join(f"{k}=={v}\n" for k, v in pins.items())
    )
    bootstrap = Path("jobs/cloud/bootstrap.py").read_text()
    bootstrap = bootstrap.replace(
        "import subprocess\n", "import subprocess\nimport sys\n"
    )
    bootstrap = bootstrap.replace(
        'os.environ["NVIDIA_TF32_OVERRIDE"] = "0"',
        'os.environ["KAGGLE_BOOTSTRAP_PYTHON"] = python\n'
        'os.environ.pop("PYTHONPATH", None)\n'
        'os.environ.pop("PYTHONHOME", None)\n'
        'os.environ["PYTHONNOUSERSITE"] = "1"\n'
        'os.environ["NVIDIA_TF32_OVERRIDE"] = "0"',
    )
    (bundle / "main.py").write_text(bootstrap)
    plan = {
        "id": f"qwen-reference-cloud-{model.split('/')[-1]}-20261020",
        "original_protocol_sha256": original_digest,
        "amendment": "Owner requested faster scoring within the remaining budget and reported $25 Cloud balance. Move the not-yet-started evaluation to CUDA L40S. Each model keeps its complete base/grid/selected-check comparison on one device. Weights, data, package versions, greedy decoding, token limits and selection rule are unchanged. No claim of exact MPS/CUDA equivalence for these adapters.",
        "budget": "One job per model, $6 reserve per job, $12 total ceiling against the owner-reported $25 available balance. Previous $30 holds remain in their original ledger pending billing; the balance report is not an invoice or a release of those holds.",
        "runtime": runtime,
        "dependency_pins": pins,
        "state": bound_state,
        "jobs": jobs,
        "tasks": tasks,
        "generation": original["generation"],
        "selection": original["selection"],
        "scope": original["scope"],
        "stopping": "One evaluation attempt per model; no training. Two-hour provider timeout. Preserve partial results, failures and all reservations. No automatic retry after generation starts.",
    }
    files = {
        p.relative_to(bundle).as_posix(): file_digest(p)
        for p in sorted(bundle.rglob("*"))
        if p.is_file()
    }
    plan.update(files_sha256=files, bundle_files_sha256=files)
    save(bundle / "experiment.json", plan)
    save(root / "protocol.json", plan)
    return root


def run(env_file):
    DESTINATION.mkdir(parents=True, exist_ok=True)
    plan = json.loads((SOURCE / "protocol.json").read_text())
    ledger = SpendingLedger(DESTINATION / "cloud-acceleration-spending.sqlite3", 12)
    provider = ServerlessJobs(ledger, DESTINATION / "jobs", cli=CLI)
    save(
        DESTINATION / "budget-basis.json",
        {
            "owner_reported_cloud_balance_usd": 25,
            "owner_reported_tf_balance_usd": 43,
            "acceleration_reserve_ceiling_usd": 12,
            "cloud_total_original_cap_usd": 30,
            "previous_cloud_ledger": "artifacts/remaining-queue-20261020/ai-cloud-spending.sqlite3",
            "previous_holds_released": False,
            "provider_billing_reconciled": False,
            "basis": "Owner's updated available-balance statement in the acceleration request; this is a separate allocation from that reported balance, not an increase in historical spending or deletion of unresolved holds.",
            "at_unix": time.time(),
        },
    )
    deadline = time.monotonic() + 2 * 3600
    submitted = {}
    while len(submitted) < 2:
        state = json.loads((SOURCE / "state.json").read_text())
        for model in MODELS:
            if model in submitted or not ready(plan, state, model):
                continue
            root = package(model)
            name = f"next-experiment-qwen-{model.split('-')[-1].lower().replace('.', '')}-eval-20261020"
            staged = stage(root, name, env_file)
            spec = JobSpec(
                name=name,
                project="project-e00d05kzpr00n7e8p79fw0",
                entry=(root / "launcher/main.py").resolve(),
                image="nvcr.io/nvidia/pytorch:25.05-py3",
                platform="gpu-l40s-a",
                preset="1gpu-8vcpu-32gb",
                bundle_sha256={"main.py": staged["launcher_sha256"]},
                output_bucket="next-experiment-artifacts",
                output_bucket_id="storagebucket-e0018072464185852935047",
                timeout_hours=2,
                disk_gib=120,
            )
            validation = provider.validate(spec)
            (root / "dry-run.json").write_text(validation)
            request = asdict(spec)
            request["entry"] = str(spec.entry)
            receipt = {
                "spec": request,
                "reservation_usd": 6,
                "reserve_basis": "Two hours plus 30 minutes startup/teardown at $1.55/L40S-hour; 120Gi disk and 20% allowance.",
                "submitted": False,
            }
            save(root / "execution-plan.json", receipt)
            job_id = provider.submit(spec, reservation_usd=6)
            receipt.update(submitted=True, job_id=job_id)
            save(root / "execution-plan.json", receipt)
            submitted[model] = job_id
            save(
                DESTINATION / "dispatch.json", {"status": "partial", "jobs": submitted}
            )
            print(model, job_id, flush=True)
        if len(submitted) == 2:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Reference training has not completed")
        time.sleep(30)
    save(DESTINATION / "dispatch.json", {"status": "complete", "jobs": submitted})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with (DESTINATION / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (DESTINATION / "dispatch-started.json").exists():
            raise FileExistsError("Inspect the previous dispatch before any restart")
        save(
            DESTINATION / "dispatch-started.json",
            {"pid": os.getpid(), "at": time.time()},
        )
        run(args.env_file)
