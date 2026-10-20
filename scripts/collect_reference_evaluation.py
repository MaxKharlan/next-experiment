"""Collect terminal Cloud reference evaluations and independently verify selection."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

from core.config import Settings
from core.connectors.compute import ServerlessJobs
from core.connectors.storage import S3Storage
from core.db.spending import SpendingLedger
from core.eval.data import read_examples
from core.eval.metrics import get_metric
from core.eval.runtime import file_digest
from core.eval.structured import TASK_EVALUATORS
from scripts.accelerate_reference_evaluation import CLI, DESTINATION, MODELS, save
from scripts.score_reference_grid import choose, verify_rows


def analyze(root, downloaded):
    plan = json.loads((root / "protocol.json").read_text())
    for name, expected in plan["files_sha256"].items():
        assert file_digest(root / "bundle" / name) == expected
    output = downloaded / "evaluation"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        for name, expected in json.loads(manifest_path.read_text()).items():
            path = Path(name)
            assert not path.is_absolute() and ".." not in path.parts
            assert file_digest(output / path) == expected
    verified = {}
    rows_total = 0
    for path in sorted((output / "scores").glob("*.json")):
        role = path.stem.rsplit("-", 1)[1]
        name = path.stem.removesuffix("-" + role)
        if name in plan["jobs"]:
            task = plan["jobs"][name]["task"]
            entry = plan["state"][name]
        else:
            key = plan["runtime"]["model"].split("/")[-1] + "-"
            assert name.startswith(key) and name.endswith("-base")
            task = name[len(key) : -len("-base")]
            entry = {}
        spec = plan["tasks"][task][role]
        examples = read_examples(root / "bundle" / spec["path"])
        record = json.loads(path.read_text())
        binding = {
            "protocol_sha256": file_digest(root / "protocol.json"),
            "original_protocol_sha256": plan["original_protocol_sha256"],
            "data_sha256": spec["sha256"],
            "runtime": plan["runtime"],
            "device": "cuda",
            "job_id": entry.get("job_id"),
            "weights_sha256": entry.get("weights_sha256"),
        }
        metric = get_metric(TASK_EVALUATORS[task][1])
        rows = verify_rows(record, examples, binding, metric)
        rows_total += len(rows)
        if not record["complete"]:
            continue
        computed = {
            "exact_match": sum(r["score"] for r in rows) / len(rows),
            "stopped_on_eos": sum(r["stopped_on_eos"] for r in rows) / len(rows),
            "hit_limit": sum(r["hit_limit"] for r in rows) / len(rows),
        }
        assert all(abs(record[k] - v) < 1e-12 for k, v in computed.items())
        verified[path.name] = computed
    results_path = output / "results.json"
    result = json.loads(results_path.read_text()) if results_path.exists() else {}
    for key, pair in result.get("pairs", {}).items():
        task = key.removeprefix(plan["runtime"]["model"].split("/")[-1] + "-")
        jobs = {n: j for n, j in plan["jobs"].items() if j["task"] == task}
        scores = {
            name: verified[name + "-validation.json"]["exact_match"] for name in jobs
        }
        assert pair["validation"] == scores
        assert pair["selected"] == choose(scores, jobs)
        assert pair["selection_rule"] == plan["selection"]
        assert pair["base"] == {
            role: verified[key + "-base-" + role + ".json"]["exact_match"]
            for role in ("validation", "check")
        }
        assert (
            pair["selected_check_exact_match"]
            == verified[pair["selected"] + "-check.json"]["exact_match"]
        )
        assert not any(
            (output / "scores" / f"{name}-check.json").exists()
            for name in jobs
            if name != pair["selected"]
        )
    if result.get("status") == "complete":
        assert manifest_path.exists()
        assert len(result["pairs"]) == 3 and len(verified) == 21
        assert rows_total == 2506
        assert json.loads((output / "preflight.json").read_text())["status"] == "passed"
        runtime = json.loads((output / "runtime.json").read_text())
        assert runtime["protocol_sha256"] == file_digest(root / "protocol.json")
        assert runtime["model"] == plan["runtime"]["model"]
        assert runtime["revision"] == plan["runtime"]["revision"]
        assert all(
            runtime["packages"][k].split("+")[0] == v
            for k, v in plan["dependency_pins"].items()
        )
    return {
        "status": result.get("status", "no_result"),
        "verified_complete_score_files": len(verified),
        "verified_rows": rows_total,
        "pairs": result.get("pairs", {}),
        "scope": plan["scope"],
    }


def collect(root, storage, provider):
    execution = json.loads((root / "execution-plan.json").read_text())
    spec = execution["spec"]
    status = provider.status(execution["job_id"])
    if status["status"]["state"] not in {"COMPLETED", "FAILED", "CANCELLED"}:
        return None
    prefix = f"runs/{spec['name']}/"
    assert status["metadata"]["name"] == spec["name"]
    assert status["metadata"]["parent_id"] == spec["project"]
    assert storage.bucket == spec["output_bucket"]
    assert any(
        v.get("source") == spec["output_bucket_id"]
        and v.get("source_path") == prefix
        and v.get("container_path") == "/outputs"
        for v in status["spec"].get("volumes", [])
    )
    destination = root / "collected"
    hashes = {}
    for key in storage.list_keys(prefix):
        assert key.startswith(prefix)
        relative = Path(key.removeprefix(prefix))
        assert not relative.is_absolute() and ".." not in relative.parts
        if relative.as_posix() == "input.zip":
            continue
        raw = storage.get_bytes(key)
        path = destination / relative
        if path.exists():
            assert path.read_bytes() == raw
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        hashes[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
    save(root / "status.json", status)
    save(
        root / "collection.json",
        {"files_sha256": hashes, "job_id": execution["job_id"]},
    )
    analysis = analyze(root, destination)
    if status["status"]["state"] == "COMPLETED":
        assert analysis["status"] == "complete"
    save(root / "analysis.json", analysis)
    return analysis


def run(env_file, wait):
    storage = S3Storage.from_settings(Settings(_env_file=env_file))
    provider = ServerlessJobs(
        SpendingLedger(DESTINATION / "cloud-acceleration-spending.sqlite3", 12),
        DESTINATION / "jobs",
        cli=CLI,
    )
    deadline = time.monotonic() + 5 * 3600
    while True:
        results = {}
        for model in MODELS:
            root = DESTINATION / model.split("/")[-1]
            if (root / "analysis.json").exists():
                results[model] = json.loads((root / "analysis.json").read_text())
            elif (root / "execution-plan.json").exists():
                execution = json.loads((root / "execution-plan.json").read_text())
                if execution.get("submitted"):
                    result = collect(root, storage, provider)
                    if result is not None:
                        results[model] = result
        save(
            DESTINATION / "collection-progress.json",
            {"at_unix": time.time(), "terminal_models": results},
        )
        if len(results) == 2 or not wait:
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Cloud reference collection deadline reached")
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    with (DESTINATION / "collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.env_file, args.wait)
