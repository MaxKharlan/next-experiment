import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api import main
from apps.api.case_input import CaseStart, prepare_case, verify_original_job
from apps.api.live_protocol import verify_datasets
from apps.api.store import InvestigationStore
from core.connectors.storage import LocalStorage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main, "store", InvestigationStore(LocalStorage(tmp_path / "store"))
    )
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    return TestClient(main.app)


@pytest.fixture
def payload(tmp_path):
    datasets = {}
    for role in ("train", "validation", "check", "test"):
        path = tmp_path / f"{role}.jsonl"
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"{role} question {i}"},
                    {"role": "assistant", "content": "card_arrival"},
                ]
            }
            for i in range(4)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        datasets[role] = {"ref": str(path), "examples": 4}
    return {
        "case": {
            "task": "banking77",
            "symptom": "The model emits incomplete intent labels.",
            "mode": "A",
            "job": {
                "id": "ftjob-explicit",
                "model": "Qwen/Qwen3-0.6B",
                "hyperparameters": {
                    "learning_rate": 0.0002,
                    "n_epochs": 3,
                    "lora_r": 16,
                    "lora_alpha": 16,
                    "batch_size": 8,
                    "context_length": 8192,
                },
            },
            "datasets": datasets,
            "evaluator": {"name": "intent_label", "metric": "exact_match"},
            "budget_usd": 0.75,
        },
        "experiment": {
            "name": "test-only",
            "noise_sd": 0.019,
            "min_effect": 0.06,
            "tokens_per_example": 57,
            "seeds": [0, 1],
        },
    }


def test_validation_does_not_start_a_worker(client, payload, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Validation must not start an investigation")

    monkeypatch.setattr(main.runners, "live", forbidden)
    response = client.post("/api/investigations/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["training_started"] is False
    assert body["provider_job_verified"] is False
    assert body["protocol"]["job_id"] == "ftjob-explicit"
    assert body["protocol"]["budget_usd"] == 0.75
    assert body["protocol"]["datasets"]["check"]["path"].endswith("check.jsonl")
    assert main.store.list() == []


def test_explicit_case_reaches_worker(client, payload, monkeypatch):
    done = threading.Event()
    seen = {}

    def live(store, inv_id, backend, *, prepared):
        seen.update(prepared)
        seen["backend"] = backend
        store.emit(inv_id, "finished", {"status": "inconclusive"})
        done.set()

    monkeypatch.setattr(main.runners, "live", live)
    response = client.post("/api/investigations/from-case", json=payload)
    assert response.status_code == 202
    assert done.wait(2)
    assert seen["backend"] == "tf"
    assert seen["protocol"]["job_id"] == "ftjob-explicit"
    assert seen["protocol"]["hyperparameters"]["learning_rate"] == 0.0002
    assert seen["agent_case"]["symptom"] == payload["case"]["symptom"]
    assert "datasets" not in seen["agent_case"]
    inv_id = response.json()["id"]
    saved = client.get(f"/api/investigations/{inv_id}").json()
    assert saved["case"]["job"]["id"] == "ftjob-explicit"


@pytest.mark.parametrize(
    "problem", ["hash", "count", "mode", "metric", "seeds", "missing"]
)
def test_invalid_case_is_rejected_before_creation(client, payload, problem):
    if problem == "hash":
        payload["case"]["datasets"]["train"]["sha256"] = "0" * 64
    elif problem == "count":
        payload["case"]["datasets"]["train"]["examples"] = 999
    elif problem == "mode":
        payload["case"]["mode"] = "B"
    elif problem == "metric":
        payload["case"]["evaluator"]["metric"] = "accuracy"
    elif problem == "seeds":
        payload["experiment"]["seeds"] = [0, 0]
    else:
        payload["case"]["datasets"]["train"]["ref"] += ".missing"
    response = client.post("/api/investigations/from-case", json=payload)
    assert response.status_code == 422
    assert main.store.list() == []


def test_public_demo_blocks_both_routes(client, payload, monkeypatch):
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(public_demo=True))
    for route in ("validate", "from-case"):
        response = client.post(f"/api/investigations/{route}", json=payload)
        assert response.status_code == 403
    assert main.store.list() == []


def test_prepared_protocol_detects_changed_files(payload):
    from pathlib import Path

    prepared = prepare_case(CaseStart.model_validate(payload))
    path = Path(payload["case"]["datasets"]["train"]["ref"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        verify_datasets(prepared["protocol"])


def test_original_job_is_checked(payload):
    prepared = prepare_case(CaseStart.model_validate(payload))
    job = SimpleNamespace(
        status="succeeded",
        model=payload["case"]["job"]["model"],
        hyperparameters=dict(payload["case"]["job"]["hyperparameters"]),
    )
    requested = []

    def get_job(job_id):
        requested.append(job_id)
        return job

    tf = SimpleNamespace(job=get_job)
    verify_original_job(tf, prepared)
    assert requested == ["ftjob-explicit"]
    job.hyperparameters["learning_rate"] = 0.00001
    with pytest.raises(ValueError, match="learning_rate"):
        verify_original_job(tf, prepared)
    job.status = "failed"
    with pytest.raises(ValueError, match="succeeded"):
        verify_original_job(tf, prepared)
