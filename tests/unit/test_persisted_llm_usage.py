from dataclasses import asdict
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api import main, runners
from apps.api.case_input import CaseStart, prepare_case
from apps.api.store import InvestigationStore
from core import config
from core.agent.router import Router
from core.agent.schemas import HypothesisList
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import (
    CallResult,
    StructuredOutputError,
    TokenFactory,
)
from core.telemetry.calls import CallLog
from tests.e2e.test_investigation_api import make_payload


class FailedProvider:
    def model(self, role):
        return "test-super"

    def structured(self, *args, **kwargs):
        raise StructuredOutputError(
            "Invalid structured response",
            [
                CallResult("test-super", "private response", 100, 20, 0.01, 10),
                CallResult("test-super", "private retry", 110, 30, 0.02, 12),
            ],
        )


def test_failed_live_run_reports_attempt_costs_after_restart(tmp_path, monkeypatch):
    payload = make_payload(tmp_path)
    prepared = prepare_case(CaseStart.model_validate(payload))
    provider = FailedProvider()
    provider.job = lambda job_id: SimpleNamespace(
        status="succeeded", **payload["case"]["job"]
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXT_EXPERIMENT_WEIGHTS", "0")
    monkeypatch.setattr(TokenFactory, "from_settings", lambda: provider)
    monkeypatch.setattr(
        config, "get_settings", lambda: SimpleNamespace(tavily_api_key=None)
    )
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    monkeypatch.setattr(
        runners, "_pair_runner", lambda *args: lambda *arms: pytest.fail("No training")
    )
    storage = LocalStorage(tmp_path / "store")
    store = InvestigationStore(storage)
    inv_id = store.create("tf", payload["case"])
    runners.live(store, inv_id, "tf", prepared=prepared)
    assert store.get(inv_id)["status"] == "failed"
    monkeypatch.setattr(main, "store", InvestigationStore(storage))
    with TestClient(main.app) as client:
        report = client.get(f"/api/investigations/{inv_id}/report").json()
    assert report["execution_status"] == "failed"
    assert "StructuredOutputError" in report["error"]["error"]
    usage = report["llm_usage"]
    assert usage["call_count"] == 2
    assert usage["cost_usd"] == pytest.approx(0.03)
    assert usage["tokens_in"] == 210 and usage["tokens_out"] == 50
    assert [c["attempt"] for c in usage["calls"]] == [1, 2]
    assert "private response" not in str(report)
    assert all("text" not in call for call in usage["calls"])


def test_call_is_persisted_before_a_terminal_event(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path)
    store = InvestigationStore(storage)
    inv_id = store.create("tf", {})
    store.emit(inv_id, "started", {})
    log = CallLog(
        inv_id, on_add=lambda call: store.emit(inv_id, "llm_call", asdict(call))
    )
    router = Router(FailedProvider(), log)
    with pytest.raises(StructuredOutputError):
        router.structured("hypotheses", [], HypothesisList, prompt_version="test-v1")
    # Simulate restart before the worker has emitted failure or completion.
    monkeypatch.setattr(main, "store", InvestigationStore(storage))
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    with TestClient(main.app) as client:
        report = client.get(f"/api/investigations/{inv_id}/report").json()
    assert report["partial"] is True
    assert report["llm_usage"]["cost_usd"] == pytest.approx(log.cost_usd)
    assert report["llm_usage"]["call_count"] == 2


def test_old_reports_have_unknown_usage(tmp_path, monkeypatch):
    store = InvestigationStore(LocalStorage(tmp_path))
    inv_id = store.create("replay", {})
    store.emit(inv_id, "finished", {"status": "inconclusive", "llm_cost_usd": 0.4})
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    with TestClient(main.app) as client:
        report = client.get(f"/api/investigations/{inv_id}/report").json()
    assert report["llm_usage"] is None
    assert report["summary"]["llm_cost_usd"] == 0.4


def test_completed_usage_is_not_added_twice_to_summary(tmp_path, monkeypatch):
    store = InvestigationStore(LocalStorage(tmp_path))
    inv_id = store.create("tf", {})
    log = CallLog(
        inv_id, on_add=lambda call: store.emit(inv_id, "llm_call", asdict(call))
    )
    log.add("verdict", "verdict-v1", CallResult("test-ultra", "text", 10, 5, 0.02, 4))
    store.emit(inv_id, "finished", {"status": "confirmed", "llm_cost_usd": 0.02})
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    with TestClient(main.app) as client:
        report = client.get(f"/api/investigations/{inv_id}/report").json()
    assert report["summary"]["llm_cost_usd"] == report["llm_usage"]["cost_usd"] == 0.02
    assert report["partial"] is False
