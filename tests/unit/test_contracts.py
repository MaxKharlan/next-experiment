import pytest
from pydantic import ValidationError

from core.agent.context import InvestigationContext
from core.agent.schemas import HypothesisList, Verdict
from core.probes.base import Probe, ProbeResult


class _DummyProbe(Probe):
    name = "dummy"
    level = 0
    outcomes = ("ok", "bad")

    def estimate_cost(self, ctx):
        return 0.0

    def run(self, ctx):
        return ProbeResult(outcome="ok", raw_ref="probes/dummy.json")


def _ctx():
    return InvestigationContext(
        investigation_id="inv-1", mode="A", job_id="ftjob-1", job_config={}
    )


def test_probe_contract():
    probe = _DummyProbe()
    assert probe.check_outcome(probe.run(_ctx())).outcome == "ok"


def test_unknown_outcome_rejected():
    with pytest.raises(ValueError):
        _DummyProbe().check_outcome(ProbeResult(outcome="maybe", raw_ref="x"))


def _hypothesis(fault_class):
    return {
        "id": "h1",
        "fault_class": fault_class,
        "location": "context_length",
        "prior": 0.4,
        "rationale": "long examples exceed the limit",
    }


def test_hypothesis_list_parses_model_output():
    parsed = HypothesisList.model_validate({"hypotheses": [_hypothesis("truncation")]})
    assert parsed.hypotheses[0].fault_class == "truncation"


def test_unknown_fault_class_rejected():
    with pytest.raises(ValidationError):
        HypothesisList.model_validate({"hypotheses": [_hypothesis("bad_vibes")]})


def test_verdict_status_is_closed_set():
    with pytest.raises(ValidationError):
        Verdict(hypothesis_id="h1", status="probably", evidence=[], patch="")
