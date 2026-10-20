import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api import code_investigations, main
from apps.api.store import InvestigationStore
from core.connectors.storage import LocalStorage
from core.db.spending import BudgetExceeded, SpendingLedger
from core.eval.runtime import PACKAGES, REQUIRED_FILES


def payload():
    return {
        "symptom": "Generation repeats the label without stopping.",
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "check": "check.jsonl",
        "code_fault": "eos_as_pad",
        "evaluation_runtime": {
            "model": "Qwen/Qwen3-0.6B",
            "revision": "a" * 40,
            "snapshot": "/snapshot/" + "a" * 40,
            "files_sha256": dict.fromkeys(REQUIRED_FILES, "b" * 64),
            "template_sha256": "c" * 64,
            "packages": dict.fromkeys(PACKAGES, "1"),
        },
    }


def test_rest_disables_paid_schema_retries(monkeypatch):
    calls = []
    monkeypatch.setattr(
        code_investigations.TokenFactory,
        "structured",
        lambda self, *args, **kwargs: calls.append(kwargs["attempts"]),
    )
    connector = object.__new__(code_investigations.SingleAttemptTokenFactory)
    connector.structured("model", [], object, attempts=3)
    assert calls == [1]


def test_code_route_is_idempotent_across_store_restart_and_forbids_changed_request(
    tmp_path, monkeypatch
):
    storage = LocalStorage(tmp_path / "storage")
    monkeypatch.setattr(main, "store", InvestigationStore(storage))
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    monkeypatch.setattr(code_investigations, "configured_ledger", lambda: object())
    prepared = []
    monkeypatch.setattr(
        code_investigations,
        "prepare",
        lambda req: prepared.append(req) or {"request": req.model_dump()},
    )
    launches = []
    monkeypatch.setattr(main, "_launch", lambda *args, **kw: launches.append(args))
    client = TestClient(main.app)
    assert client.post("/api/investigations/code", json=payload()).status_code == 422
    headers = {"Idempotency-Key": "code-study-1"}
    first = client.post("/api/investigations/code", json=payload(), headers=headers)
    assert first.status_code == 202, first.text
    monkeypatch.setattr(main, "store", InvestigationStore(storage))
    repeat = client.post("/api/investigations/code", json=payload(), headers=headers)
    assert repeat.json() == {"id": first.json()["id"], "reused": True}
    assert len(launches) == len(prepared) == 1
    modified = {**payload(), "code_fault": "double_shift"}
    assert (
        client.post(
            "/api/investigations/code", json=modified, headers=headers
        ).status_code
        == 422
    )
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(public_demo=True))
    assert (
        client.post(
            "/api/investigations/code", json=payload(), headers=headers
        ).status_code
        == 403
    )


def test_invalid_seed_types_and_duplicate_seeds_fail_before_dispatch():
    for seeds in ([True, 2], [1, 1], [-1, 2], ["1", 2], None, 3, [[1], [2]]):
        with pytest.raises(ValueError):
            code_investigations.CodeInvestigationStart.model_validate(
                {**payload(), "seeds": seeds}
            )


def test_global_ledger_adds_allowance_and_retains_pending_call(tmp_path):
    ledger = SpendingLedger(tmp_path / "spend.db", 25)
    guarded = code_investigations.AllowanceLedger(ledger)
    guarded.reserve("first", "review", 1)
    assert ledger.snapshot()["committed_usd"] == pytest.approx(1.2)
    guarded.reserve("second", "ambiguous", 2)
    guarded.settle("first", 0.5)
    assert ledger.snapshot()["committed_usd"] == pytest.approx(3)


def test_request_cap_stops_paid_dispatch_even_when_global_budget_remains(tmp_path):
    global_ledger = SpendingLedger(tmp_path / "global.db", 25)
    request = SpendingLedger(tmp_path / "request.db", 1)
    guarded = code_investigations.AllowanceLedger(global_ledger, request)
    guarded.reserve("first", "review", 0.5)
    with pytest.raises(BudgetExceeded):
        guarded.reserve("second", "too much", 0.5)
    assert global_ledger.snapshot()["committed_usd"] == pytest.approx(0.6)
    guarded.settle("first", 0.1)
    assert request.snapshot()["committed_usd"] == pytest.approx(0.12)


def test_model_lock_prevents_concurrent_workers_and_releases_after_failure(tmp_path):
    lock = tmp_path / "model.lock"
    with pytest.raises(RuntimeError), code_investigations.model_execution_lock(lock):
        with (
            pytest.raises(ValueError, match="Another"),
            code_investigations.model_execution_lock(lock),
        ):
            pytest.fail("Concurrent model work was admitted")
        raise RuntimeError("training failed")
    with code_investigations.model_execution_lock(lock):
        pass


def test_source_changes_fail_before_execution(tmp_path, monkeypatch):
    source = tmp_path / "trainer.py"
    source.write_text("original")
    monkeypatch.setattr(code_investigations, "verify_datasets", lambda value: None)
    monkeypatch.setattr(code_investigations, "verify_runtime", lambda *a, **k: None)
    prepared = {
        "model": "Qwen/Qwen3-0.6B",
        "evaluation_runtime": {},
        "source_sha256": {str(source): code_investigations.file_digest(source)},
    }
    code_investigations.verify_prepared(prepared)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        code_investigations.verify_prepared(prepared)


def test_context_guard_counts_batch_encoding_tokens(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    monkeypatch.setattr(code_investigations, "verify_runtime", lambda *a, **kw: None)
    paths = {}
    for role in ("train", "validation", "check"):
        path = tmp_path / f"{role}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": role},
                        {"role": "assistant", "content": "label"},
                    ]
                }
            )
            + "\n"
        )
        paths[role] = str(path)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(
            apply_chat_template=lambda *a, **kw: {
                "input_ids": [1] * 513,
                "attention_mask": [1] * 513,
            }
        ),
    )
    request = code_investigations.CodeInvestigationStart.model_validate(
        {**payload(), **paths}
    )
    with pytest.raises(ValueError, match="would be truncated"):
        code_investigations.prepare(request)
