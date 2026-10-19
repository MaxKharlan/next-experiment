import json
from dataclasses import asdict

import httpx
import pytest
from openai import APIStatusError, OpenAI

from apps.api.store import InvestigationStore
from core.agent.router import Router
from core.agent.schemas import HypothesisList
from core.config import Settings
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TokenFactory
from core.db.spending import SpendingLedger
from core.report.investigation import build_report
from core.telemetry.calls import CallLog, LLMCall
from tests.unit.test_token_factory import VALID


def provider(tmp_path, handler, *, spending=True):
    client = OpenAI(
        api_key="test",
        base_url="https://provider.test/v1/",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    ledger = SpendingLedger(tmp_path / "budget.sqlite3", 5) if spending else None
    return TokenFactory(client, Settings(_env_file=None), spending=ledger), ledger


def response(
    content="ok", *, request_id="request-1", response_id="completion-1", usage=True
):
    body = {
        "id": response_id,
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if usage:
        body["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }
    return httpx.Response(
        200, json=body, headers={"x-request-id": request_id} if request_id else {}
    )


def test_provider_ids_survive_call_log_storage_and_report_restart(tmp_path):
    tf, ledger = provider(tmp_path, lambda _: response())
    storage = LocalStorage(tmp_path / "events")
    store = InvestigationStore(storage)
    inv = store.create("tf", {})
    log = CallLog(inv, on_add=lambda call: store.emit(inv, "llm_call", asdict(call)))
    Router(tf, log).chat("hypotheses", [], prompt_version="test")
    call = log.calls[0]
    assert call.provider_request_id == "request-1"
    assert call.provider_response_id == "completion-1"
    snapshot = ledger.snapshot()
    assert call.spending_operation_id == snapshot["operations"][0]["id"]
    assert (
        snapshot["provider_receipts"][0]["operation_id"] == call.spending_operation_id
    )
    assert snapshot["operations"][0]["status"] == "settled"
    saved = json.loads(storage.get_text(log.save(storage)))
    assert saved["provider_request_id"] == "request-1"
    report = build_report(InvestigationStore(storage).snapshot(inv))
    assert report["llm_usage"]["calls"][0]["provider_response_id"] == "completion-1"
    assert "text" not in report["llm_usage"]["calls"][0]


def test_structured_retries_keep_distinct_provider_ids_and_reservations(tmp_path):
    requests = []

    def respond(request):
        requests.append(request)
        n = len(requests)
        return response(
            "{}" if n == 1 else VALID, request_id=f"req-{n}", response_id=f"cmp-{n}"
        )

    tf, ledger = provider(tmp_path, respond)
    log = CallLog("test")
    Router(tf, log).structured("hypotheses", [], HypothesisList, prompt_version="test")
    assert [c.provider_request_id for c in log.calls] == ["req-1", "req-2"]
    assert [c.provider_response_id for c in log.calls] == ["cmp-1", "cmp-2"]
    assert len({c.spending_operation_id for c in log.calls}) == 2
    assert len(ledger.snapshot()["provider_receipts"]) == 2


def test_missing_ids_and_legacy_logs_are_unknown_not_fabricated(tmp_path):
    tf, _ = provider(
        tmp_path, lambda _: response(request_id=None, response_id=None), spending=False
    )
    call = tf.chat(tf.model("super"), [])
    assert call.provider_request_id is None and call.provider_response_id is None
    assert call.spending_operation_id is None
    old = asdict(CallLog("old").add("hypotheses", "v1", call))
    for field in [
        "provider_request_id",
        "provider_response_id",
        "spending_operation_id",
    ]:
        old.pop(field)
    assert LLMCall(**old).provider_request_id is None


def test_http_error_reference_is_durable_and_hold_remains(tmp_path):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            502,
            headers={"x-request-id": "failed-request"},
            json={"error": {"message": "ambiguous"}},
        )

    tf, ledger = provider(tmp_path, respond)
    with pytest.raises(APIStatusError):
        tf.chat(tf.model("super"), [])
    saved = SpendingLedger(ledger.path, 5).snapshot()
    assert len(requests) == 1
    assert saved["operations"][0]["status"] == "pending"
    assert saved["provider_receipts"][0]["provider_request_id"] == "failed-request"
    assert saved["provider_receipts"][0]["outcome"] == "http_error"
    assert saved["provider_receipts"][0]["http_status"] == 502


def test_missing_usage_keeps_response_ids_without_settling(tmp_path):
    tf, ledger = provider(tmp_path, lambda _: response(usage=False))
    with pytest.raises(RuntimeError, match="omitted usage"):
        tf.chat(tf.model("super"), [])
    snapshot = SpendingLedger(ledger.path, 5).snapshot()
    assert snapshot["operations"][0]["status"] == "pending"
    assert snapshot["provider_receipts"][0]["provider_response_id"] == "completion-1"


def test_receipt_storage_failure_stops_structured_retries(tmp_path, monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return response("{}")

    tf, ledger = provider(tmp_path, respond)

    def fail(*args, **kwargs):
        raise OSError("receipt storage failed")

    monkeypatch.setattr(ledger, "record_provider_receipt", fail)
    with pytest.raises(OSError, match="receipt storage failed"):
        tf.structured(tf.model("super"), [], HypothesisList)
    assert len(requests) == 1
    assert ledger.snapshot()["operations"][0]["status"] == "pending"


def test_receipt_is_immutable_and_migrates_legacy_ledger(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE budget (id INTEGER PRIMARY KEY, cap REAL NOT NULL)")
        db.execute("INSERT INTO budget VALUES (1,5)")
        db.execute(
            "CREATE TABLE spending (id TEXT PRIMARY KEY, purpose TEXT NOT NULL, reserved REAL NOT NULL, actual REAL, status TEXT NOT NULL)"
        )
        db.execute("INSERT INTO spending VALUES ('op','inference',0.2,NULL,'pending')")
    ledger = SpendingLedger(path, 5)
    kwargs = {
        "request_id": "req",
        "response_id": "cmp",
        "outcome": "response",
        "http_status": None,
    }
    ledger.record_provider_receipt("op", **kwargs)
    ledger.record_provider_receipt("op", **kwargs)
    with pytest.raises(ValueError, match="conflicts"):
        ledger.record_provider_receipt("op", **{**kwargs, "request_id": "different"})
    with pytest.raises(ValueError, match="existing reservation"):
        ledger.record_provider_receipt("unknown", **kwargs)
    assert ledger.snapshot()["committed_usd"] == 0.2
    assert len(ledger.snapshot()["provider_receipts"]) == 1
