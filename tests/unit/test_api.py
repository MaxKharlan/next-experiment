import json
import time

import pytest
from fastapi.testclient import TestClient

from apps.api import main
from apps.api.store import InvestigationStore
from core.connectors.storage import LocalStorage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", InvestigationStore(LocalStorage(tmp_path)))
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
    assert "verdict" in kinds
    assert "event: finished" in body
    listed = client.get("/api/investigations").json()
    assert listed[0]["id"] == inv_id and listed[0]["status"] == "finished"


def test_finished_investigation_survives_a_restart(client, tmp_path):
    inv_id = client.post(
        "/api/investigations", json={"backend": "replay", "delay_s": 0}
    ).json()["id"]
    _wait(client, inv_id)
    fresh = InvestigationStore(LocalStorage(tmp_path))
    assert fresh.get(inv_id)["status"] == "finished"
    assert fresh.events_since(inv_id, 0)[-1]["type"] == "finished"


def test_public_demo_only_replays(client, monkeypatch):
    monkeypatch.setattr(
        main, "get_settings", lambda: type("S", (), {"public_demo": True})()
    )
    response = client.post("/api/investigations", json={"backend": "tf"})
    assert response.status_code == 403
    assert client.get("/api/investigations/nope").status_code == 404
