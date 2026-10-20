import hashlib
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.connectors.compute import JobSpec, ServerlessJobs
from core.db.spending import BudgetExceeded, SpendingLedger


def spec(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    entry = directory / "main.py"
    entry.write_text("print('bounded job')\n")
    return JobSpec(
        "test-cloud-one",
        "project-test",
        entry,
        "example/runtime:fixed",
        "gpu-l40s-a",
        "1gpu-8vcpu-32gb",
        {"main.py": hashlib.sha256(entry.read_bytes()).hexdigest()},
        output_bucket="test-output",
        output_bucket_id="storagebucket-test",
    )


def test_submit_is_durable_idempotent_and_reserves_before_cli(tmp_path):
    job = spec(tmp_path)
    ledger = SpendingLedger(tmp_path / "spending.db", 5)
    calls = []

    def run(args, **kwargs):
        assert ledger.snapshot()["committed_usd"] == 4
        assert "--detach" in args and args[args.index("--timeout") + 1] == "1h"
        assert "--output" in args and "--env" not in args
        calls.append(args)
        return SimpleNamespace(
            returncode=0, stdout='{"resource_id":"aijob-test"}', stderr=""
        )

    client = ServerlessJobs(ledger, tmp_path / "receipts", run=run)
    assert client.submit(job, reservation_usd=4) == "aijob-test"
    client = ServerlessJobs(ledger, tmp_path / "receipts", run=run)
    assert client.submit(job, reservation_usd=4) == "aijob-test"
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        client.submit(replace(job, timeout_hours=2), reservation_usd=4)


def test_ambiguous_submission_keeps_reservation_and_cannot_retry(tmp_path):
    job = spec(tmp_path)
    ledger = SpendingLedger(tmp_path / "spending.db", 5)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args, 120)

    client = ServerlessJobs(ledger, tmp_path / "receipts", run=run)
    with pytest.raises(subprocess.TimeoutExpired):
        client.submit(job, reservation_usd=4)
    with pytest.raises(ValueError, match="Ambiguous"):
        client.submit(job, reservation_usd=4)
    assert ledger.snapshot()["committed_usd"] == 4
    assert len(calls) == 1


def test_budget_and_bundle_drift_stop_before_dispatch(tmp_path):
    job = spec(tmp_path)
    ledger = SpendingLedger(tmp_path / "spending.db", 3)
    client = ServerlessJobs(
        ledger, tmp_path / "receipts", run=lambda *a, **k: pytest.fail("dispatch")
    )
    with pytest.raises(BudgetExceeded):
        client.submit(job, reservation_usd=4)
    job.entry.write_text("different\n")
    with pytest.raises(ValueError, match="manifest"):
        client.submit(job, reservation_usd=1)
    assert ledger.snapshot()["committed_usd"] == 0


def test_hidden_files_and_symlinks_are_never_packaged(tmp_path):
    job = spec(tmp_path)
    hidden = job.entry.parent / ".env"
    hidden.write_text("secret")
    with pytest.raises(ValueError, match="Hidden"):
        job.validate()
    hidden.unlink()
    (job.entry.parent / "credentials").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="Symlinks"):
        job.validate()


def test_results_require_completion_identity_bucket_and_digest(tmp_path):
    job = spec(tmp_path)
    state = {
        "metadata": {"name": job.name, "parent_id": job.project},
        "status": {"state": "RUNNING"},
        "spec": {
            "volumes": [
                {
                    "source": job.output_bucket_id,
                    "source_path": f"runs/{job.name}/",
                    "container_path": "/outputs",
                }
            ]
        },
    }
    client = ServerlessJobs(
        SpendingLedger(tmp_path / "spend.db", 5),
        tmp_path / "receipts",
        run=lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=json.dumps(state), stderr=""
        ),
    )
    raw = b'{"score":0.5}'
    keys = []
    storage = SimpleNamespace(
        bucket="test-output", get_bytes=lambda key: keys.append(key) or raw
    )
    with pytest.raises(ValueError, match="completed"):
        client.fetch_json(storage, job, "aijob-test", "result.json")
    state["status"]["state"] = "COMPLETED"
    assert client.fetch_json(
        storage,
        job,
        "aijob-test",
        "result.json",
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    ) == {"score": 0.5}
    assert keys == ["runs/test-cloud-one/result.json"]
    with pytest.raises(ValueError, match="digest"):
        client.fetch_json(
            storage, job, "aijob-test", "result.json", expected_sha256="bad"
        )
    with pytest.raises(ValueError, match="relative"):
        client.fetch_json(storage, job, "aijob-test", "../result.json")
    state["metadata"]["name"] = "another-job"
    with pytest.raises(ValueError, match="namespace"):
        client.fetch_json(storage, job, "aijob-test", "result.json")
    state["metadata"]["name"] = job.name
    state["metadata"]["parent_id"] = "project-another"
    with pytest.raises(ValueError, match="namespace"):
        client.fetch_json(storage, job, "aijob-test", "result.json")
    state["metadata"]["parent_id"] = job.project
    state["spec"]["volumes"][0]["source"] = "storagebucket-another"
    before = len(keys)
    with pytest.raises(ValueError, match="output mount"):
        client.fetch_json(storage, job, "aijob-test", "result.json")
    assert len(keys) == before
