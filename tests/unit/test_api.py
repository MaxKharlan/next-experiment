import json
import time
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from apps.api import main
from apps.api.store import InvestigationStore
from core.connectors.storage import LocalStorage
from core.db.investigations import SQLiteInvestigations
from core.interventions.pairs import Intervention
from core.interventions.retrain import plan_retrain


@pytest.fixture(params=["storage", "sqlite"])
def client(tmp_path, monkeypatch, request):
    repository = (
        SQLiteInvestigations(f"sqlite:///{tmp_path}/investigations.db")
        if request.param == "sqlite"
        else None
    )
    monkeypatch.setattr(
        main, "store", InvestigationStore(LocalStorage(tmp_path), repository)
    )
    return TestClient(main.app)


def _wait(client, inv_id):
    for _ in range(100):
        if client.get(f"/api/investigations/{inv_id}").json()["status"] == "finished":
            return
        time.sleep(0.05)
    raise AssertionError("replay did not finish")


def test_screen_and_demo_cases(client):
    assert "Next Experiment" in client.get("/").text
    names = [c["name"] for c in client.get("/api/demo-cases").json()]
    assert "fault6-tf" in names


def test_replay_streams_the_saved_investigation(client):
    inv_id = client.post(
        "/api/investigations", json={"backend": "replay", "delay_s": 0}
    ).json()["id"]
    _wait(client, inv_id)
    with client.stream("GET", f"/api/investigations/{inv_id}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    kinds = [
        json.loads(line[6:]).get("kind")
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert kinds[1:4] == ["hypotheses", "probe", "intervention"]
    assert kinds[4:8] == ["verdict", "retrain_proposal", "retrain", "comparison"]
    assert "event: finished" in body
    listed = client.get("/api/investigations").json()
    assert listed[0]["id"] == inv_id and listed[0]["status"] == "finished"


def test_finished_investigation_survives_a_restart(client, tmp_path):
    inv_id = client.post(
        "/api/investigations", json={"backend": "replay", "delay_s": 0}
    ).json()["id"]
    _wait(client, inv_id)
    fresh = InvestigationStore(LocalStorage(tmp_path), main.store.repository)
    assert fresh.get(inv_id)["status"] == "finished"
    assert fresh.events_since(inv_id, 0)[-1]["type"] == "finished"


def test_public_demo_only_replays(client, monkeypatch):
    monkeypatch.setattr(
        main, "get_settings", lambda: type("S", (), {"public_demo": True})()
    )
    response = client.post("/api/investigations", json={"backend": "tf"})
    assert response.status_code == 403
    assert client.get("/api/investigations/nope").status_code == 404


PLAN = asdict(
    plan_retrain(
        Intervention("h1", "target", {"learning_rate": 1e-4}, "lr up"),
        {"learning_rate": 1e-5, "n_epochs": 3},
        "Qwen/Qwen3-0.6B",
        200,
        57,
        "ftjob-c0",
    )
)
SUMMARY = {"status": "confirmed", "llm_cost_usd": 0.007, "llm_calls": {}}


def _wait_for(client, inv_id, status):
    for _ in range(100):
        if client.get(f"/api/investigations/{inv_id}").json()["status"] == status:
            return
        time.sleep(0.05)
    raise AssertionError(f"{inv_id} never reached {status}")


def _waiting_run(store, inv_id, backend):
    store.emit(inv_id, "started", {"case": {}, "backend": backend})
    store.emit(inv_id, "step", {"kind": "retrain_proposal", **PLAN})
    store.emit(inv_id, "awaiting_approval", {"plan": PLAN, "summary": SUMMARY})


def _events(client, inv_id):
    return main.store.events_since(inv_id, 0)


def test_approved_retrain_runs_once_and_finishes(client, monkeypatch):
    from core.connectors.token_factory import TokenFactory
    from scripts import retrain_tf

    plans = []

    def fake_retrainer(tf, storage, inv_id):
        def run(plan, on_step):
            plans.append(plan)
            on_step({"kind": "retrain", "jobs": {"fixed": "ftjob-9"}})
            return {"kind": "comparison", "gain": 0.37}

        return run

    monkeypatch.setattr(main.runners, "live", _waiting_run)
    monkeypatch.setattr(retrain_tf, "make_retrainer", fake_retrainer)
    monkeypatch.setattr(TokenFactory, "from_settings", lambda: None)
    inv_id = client.post("/api/investigations", json={"backend": "tf"}).json()["id"]
    _wait_for(client, inv_id, "awaiting_approval")

    answer = client.post(
        f"/api/investigations/{inv_id}/retrain", json={"approve": True}
    )
    assert answer.status_code == 200
    again = client.post(f"/api/investigations/{inv_id}/retrain", json={"approve": True})
    assert again.status_code == 409
    _wait_for(client, inv_id, "finished")

    assert len(plans) == 1 and plans[0].failed_job_id == "ftjob-c0"
    types = [(e["type"], e["data"].get("kind")) for e in _events(client, inv_id)]
    assert types[-4:] == [
        ("resumed", None),
        ("step", "retrain"),
        ("step", "comparison"),
        ("finished", None),
    ]


def test_declined_retrain_finishes_without_training(client, monkeypatch):
    monkeypatch.setattr(main.runners, "live", _waiting_run)
    inv_id = client.post("/api/investigations", json={"backend": "tf"}).json()["id"]
    _wait_for(client, inv_id, "awaiting_approval")
    client.post(f"/api/investigations/{inv_id}/retrain", json={"approve": False})
    _wait_for(client, inv_id, "finished")
    last = _events(client, inv_id)[-2:]
    assert last[0]["data"] == {"kind": "retrain_declined"}
    assert last[1]["data"] == SUMMARY


def test_waiting_investigation_survives_a_restart(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main.runners, "live", _waiting_run)
    inv_id = client.post("/api/investigations", json={"backend": "tf"}).json()["id"]
    _wait_for(client, inv_id, "awaiting_approval")
    fresh = InvestigationStore(LocalStorage(tmp_path), main.store.repository)
    assert fresh.get(inv_id)["status"] == "awaiting_approval"
    assert fresh.resume(inv_id, False)["plan"] == PLAN


def test_public_demo_does_not_retrain(client, monkeypatch):
    monkeypatch.setattr(
        main, "get_settings", lambda: type("S", (), {"public_demo": True})()
    )
    response = client.post("/api/investigations/x/retrain", json={"approve": True})
    assert response.status_code == 403
