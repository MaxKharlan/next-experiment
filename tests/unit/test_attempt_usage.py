from dataclasses import asdict

import httpx
import pytest

from apps.api.store import InvestigationStore
from core.agent.router import Router
from core.agent.schemas import HypothesisList
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import StructuredOutputError
from core.telemetry.calls import CallLog
from tests.unit.test_token_factory import VALID, FakeCompletions, _tf


@pytest.mark.parametrize(
    "responses,error",
    [
        (["{}", VALID], None),
        (["{}", "{}", "{}"], StructuredOutputError),
        (["{}", httpx.ReadTimeout("transport interrupted")], httpx.ReadTimeout),
    ],
)
def test_completed_attempts_are_persisted_before_next_request(
    tmp_path, responses, error
):
    storage = LocalStorage(tmp_path)
    store = InvestigationStore(storage)
    inv_id = store.create("tf", {})
    store.emit(inv_id, "started", {})
    log = CallLog(
        inv_id, on_add=lambda call: store.emit(inv_id, "llm_call", asdict(call))
    )

    class ObservedCompletions(FakeCompletions):
        def create(self, **kwargs):
            # Before each request, every earlier response is already durable.
            saved = store.snapshot(inv_id)["events"]
            calls = [e for e in saved if e["type"] == "llm_call"]
            assert len(calls) == len(self.calls)
            return super().create(**kwargs)

    fake = ObservedCompletions(responses)
    router = Router(_tf(fake), log)
    if error:
        with pytest.raises(error):
            router.structured("hypotheses", [], HypothesisList, prompt_version="test")
    else:
        router.structured("hypotheses", [], HypothesisList, prompt_version="test")
    expected = len(responses) - (1 if error is httpx.ReadTimeout else 0)
    assert len(log.calls) == expected
    assert [c.attempt for c in log.calls] == list(range(1, expected + 1))
    restarted = InvestigationStore(storage)
    saved = [
        e["data"] for e in restarted.events_since(inv_id, 0) if e["type"] == "llm_call"
    ]
    assert len(saved) == expected
    assert sum(c["cost_usd"] for c in saved) == pytest.approx(log.cost_usd)
    assert sum(c["tokens_in"] for c in saved) == expected * 1000


def test_storage_callback_failure_stops_before_retry():
    def fail(call):
        raise OSError("storage unavailable")

    fake = FakeCompletions(["{}", VALID])
    log = CallLog("test", on_add=fail)
    router = Router(_tf(fake), log)
    with pytest.raises(OSError, match="storage unavailable"):
        router.structured("hypotheses", [], HypothesisList, prompt_version="test")
    assert len(fake.calls) == 1
    assert len(log.calls) == 1
