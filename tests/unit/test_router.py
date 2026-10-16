import json

import pytest

from core.agent.router import Router
from core.agent.schemas import HypothesisList
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import CallResult, StructuredOutputError
from core.telemetry.calls import CallLog

PARSED = HypothesisList.model_validate(
    {
        "hypotheses": [
            {
                "id": "h1",
                "fault_class": "lr_too_low",
                "location": "config",
                "prior": 0.7,
                "rationale": "",
            }
        ]
    }
)


def _call(model, cost=0.001):
    return CallResult(model, "{}", 100, 50, cost, 200)


class FakeTF:
    def __init__(self, fail=False):
        self.fail = fail

    def model(self, role):
        return f"nemotron-{role}"

    def chat(self, model, messages, **kwargs):
        return _call(model)

    def structured(self, model, messages, schema, **kwargs):
        attempts = [_call(model), _call(model)]
        if self.fail:
            raise StructuredOutputError("no valid JSON", attempts)
        return PARSED, attempts


def test_steps_go_to_their_model():
    router = Router(FakeTF(), CallLog("inv-1"))
    assert router.model("audit_example") == "nemotron-nano"
    assert router.model("hypotheses") == "nemotron-super"
    assert router.model("verdict") == "nemotron-ultra"
    with pytest.raises(KeyError):
        router.model("guess")


def test_super_steps_move_to_ultra_after_two_mismatches():
    router = Router(FakeTF(), CallLog("inv-1"))
    router.mismatch()
    assert router.role("hypotheses") == "super"
    router.mismatch()
    assert router.role("hypotheses") == "ultra"
    assert router.role("known_issues") == "super"
    assert router.role("audit_example") == "nano"


def test_every_call_and_retry_is_logged():
    log = CallLog("inv-1")
    router = Router(FakeTF(), log)
    router.chat("audit_example", [], prompt_version="audit-v1")
    assert router.structured("hypotheses", [], HypothesisList, prompt_version="hyp-v1")
    assert [(c.purpose, c.attempt) for c in log.calls] == [
        ("audit_example", 1),
        ("hypotheses", 1),
        ("hypotheses", 2),
    ]
    by_model = log.totals("model")
    assert by_model["nemotron-super"]["calls"] == 2
    assert log.cost_usd == pytest.approx(0.003)


def test_failed_structured_call_still_costs():
    log = CallLog("inv-1")
    router = Router(FakeTF(fail=True), log)
    with pytest.raises(StructuredOutputError):
        router.structured("verdict", [], HypothesisList, prompt_version="v1")
    assert len(log.calls) == 2
    assert log.totals("purpose")["verdict"]["tokens_in"] == 200


def test_log_is_saved_as_jsonl(tmp_path):
    log = CallLog("inv-1")
    Router(FakeTF(), log).chat("verdict", [], prompt_version="verdict-v1")
    key = log.save(LocalStorage(tmp_path))
    rows = [json.loads(line) for line in (tmp_path / key).read_text().splitlines()]
    assert rows[0]["model"] == "nemotron-ultra"
    assert rows[0]["prompt_version"] == "verdict-v1"
