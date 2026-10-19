import json

from core.agent.context import InvestigationContext
from core.agent.loop import investigate
from core.agent.schemas import (
    Hypothesis,
    HypothesisList,
    ProbeInterpretation,
    Verdict,
)
from core.connectors.storage import LocalStorage
from core.telemetry.calls import CallLog

JOB_HP = {
    "learning_rate": 1e-5,
    "n_epochs": 3,
    "lora_r": 16,
    "lora_alpha": 16,
    "batch_size": 8,
}
CHECKPOINTS = [
    {"train_loss": 3.25, "valid_loss": 2.79},
    {"train_loss": 2.66, "valid_loss": 2.51},
    {"train_loss": 2.46, "valid_loss": 2.45},
]


def _hyp(id_, fault_class, prior):
    return Hypothesis(
        id=id_, fault_class=fault_class, location="config", prior=prior, rationale=""
    )


class FakeRouter:
    def __init__(self, hypotheses):
        self.log = CallLog("inv-1")
        self.hypotheses = hypotheses
        self.steps = []

    def structured(self, step, messages, schema, *, prompt_version):
        self.steps.append(step)
        if step == "hypotheses":
            return HypothesisList(hypotheses=self.hypotheses)
        if step == "interpret_probe":
            return ProbeInterpretation(probe_name="", outcome="", explanation="card")
        return Verdict(
            hypothesis_id="", status="rejected", evidence=[], patch="", summary="ok"
        )


def _ctx(checkpoints=CHECKPOINTS, **hp):
    return InvestigationContext(
        "inv-1",
        "A",
        "ftjob-1",
        {"hyperparameters": {**JOB_HP, **hp}, "checkpoints": checkpoints},
        budget_left_usd=1.0,
    )


def _runner(results):
    """Fills metrics from a table: fault class of the pair -> (target, control)."""
    calls = []

    def run_pair(target, control):
        calls.append(target.patch)
        target.metrics, control.metrics = results[target.hypothesis_id]

    return run_pair, calls


def test_fault6_is_found_and_confirmed(tmp_path):
    router = FakeRouter(
        [_hyp("h1", "lr_too_low", 0.4), _hyp("h2", "chat_template_mismatch", 0.3)]
    )
    run_pair, calls = _runner({"h1": ([0.16, 0.20], [0.02, 0.02])})
    result = investigate(_ctx(), {"symptom": "base-like"}, router, run_pair)

    assert result.status == "confirmed"
    assert calls == [{"learning_rate": 1e-4}]
    kinds = [s["kind"] for s in result.steps]
    assert kinds[0] == "hypotheses" and kinds[-2:] == ["intervention", "verdict"]
    assert "probe" in kinds
    assert result.verdict.status == "confirmed"
    assert result.verdict.hypothesis_id == "h1"
    assert "intervention:h1" in result.verdict.evidence
    assert router.steps[0] == "hypotheses" and router.steps[-1] == "verdict"
    key = result.save(LocalStorage(tmp_path))
    assert json.loads((tmp_path / key).read_text())["status"] == "confirmed"


def test_rejected_leader_hands_over_to_the_next():
    router = FakeRouter(
        [_hyp("h1", "too_many_epochs", 0.5), _hyp("h2", "lr_too_low", 0.3)]
    )
    run_pair, calls = _runner(
        {"h1": ([0.12, 0.12], [0.46, 0.42]), "h2": ([0.16, 0.20], [0.02, 0.02])}
    )
    # config and curve point at epochs, so they are tested first and rejected; the
    # low rate keeps h2 likely enough to be tested next
    overfit = [
        {"train_loss": t, "valid_loss": v}
        for t, v in [(1.17, 0.45), (0.33, 1.03), (0.08, 2.03)]
    ]
    ctx = _ctx(overfit, n_epochs=12, learning_rate=2e-5)
    result = investigate(ctx, {}, router, run_pair, explain=False)
    assert [list(c) for c in calls] == [["n_epochs"], ["learning_rate"]]
    assert result.status == "confirmed"
    assert result.verdict.hypothesis_id == "h2"
    assert "interpret_probe" not in router.steps


def test_stops_when_nothing_is_testable():
    router = FakeRouter([_hyp("h1", "chat_template_mismatch", 0.8)])
    run_pair, calls = _runner({})
    result = investigate(_ctx(), {}, router, run_pair, explain=False)
    assert result.status == "inconclusive"
    assert calls == []
    assert result.steps[-1]["kind"] == "stop"


class FakeTavily:
    def __init__(self):
        self.spent = {}

    def search(self, query, *, investigation_id):
        self.spent[investigation_id] = self.spent.get(investigation_id, 0) + 1
        hit = {"title": "t", "url": "https://a.example/lr", "content": "c", "score": 1}
        return {"results": [hit]}


def test_known_issues_go_into_the_case_and_the_report():
    router = FakeRouter([_hyp("h1", "lr_too_low", 0.6)])
    run_pair, _ = _runner({"h1": ([0.16, 0.20], [0.02, 0.02])})
    result = investigate(
        _ctx(), {"symptom": "base-like"}, router, run_pair, tavily=FakeTavily()
    )
    known = result.steps[0]
    assert known["kind"] == "known_issues"
    assert known["sources"][0]["url"] == "https://a.example/lr"
    assert known["credits"] == 2
    assert result.steps[1]["kind"] == "hypotheses"


def test_steps_are_streamed_as_they_happen():
    router = FakeRouter([_hyp("h1", "lr_too_low", 0.6)])
    run_pair, _ = _runner({"h1": ([0.16, 0.20], [0.02, 0.02])})
    seen = []
    result = investigate(
        _ctx(), {}, router, run_pair, explain=False, on_step=seen.append
    )
    assert seen == result.steps
    assert seen[0]["hypotheses"][0]["fault_class"] == "lr_too_low"


def test_confirmed_fix_ends_with_the_retrain_proposal():
    router = FakeRouter([_hyp("h1", "lr_too_low", 0.6)])
    run_pair, _ = _runner({"h1": ([0.16, 0.20], [0.02, 0.02])})
    asked = []

    def propose(target, control):
        asked.append((target.patch, control.patch))
        return {"cost_usd": 0.014}

    result = investigate(
        _ctx(), {}, router, run_pair, explain=False, propose_retrain=propose
    )
    assert asked == [({"learning_rate": 1e-4}, {})]
    assert result.steps[-1] == {"kind": "retrain_proposal", "cost_usd": 0.014}
    assert json.loads(result.to_json())["retrain_plan"] == {"cost_usd": 0.014}


def test_no_proposal_without_a_confirmed_fix():
    router = FakeRouter([_hyp("h1", "chat_template_mismatch", 0.8)])
    run_pair, _ = _runner({})
    result = investigate(
        _ctx(), {}, router, run_pair, explain=False, propose_retrain=lambda t, c: {}
    )
    assert result.retrain_plan is None
    assert "retrain_proposal" not in [s["kind"] for s in result.steps]


def test_mode_b_label_checks_reach_hypotheses_without_raw_tokens(monkeypatch):
    from copy import deepcopy

    from core.agent import loop

    ctx = _ctx()
    ctx.mode = "B"
    rows = [{"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, 2]}]
    case = {"training_labels": {"reference": rows, "observed": rows}}
    original = deepcopy(case)
    captured = []
    hypotheses = [_hyp("h1", "loss_on_prompt", 0.4)]

    def propose(router, supplied_case, **kwargs):
        captured.append(supplied_case)
        return hypotheses

    monkeypatch.setattr(loop, "propose_hypotheses", propose)
    recorded = []
    result = investigate(
        ctx,
        case,
        FakeRouter(hypotheses),
        lambda *_: None,
        probes=(),
        causal_checks=False,
        on_step=recorded.append,
    )
    assert case == original
    assert "training_labels" not in captured[0]
    assert captured[0]["training_label_checks"]["alignment"]["status"] == "aligned"
    assert result.steps[0]["kind"] == "training_label_checks"
    assert recorded[0] == result.steps[0]
    assert result.status == "inconclusive"
    assert result.hypotheses[0].prior == 0.4


def test_mode_a_rejects_label_attachment_before_model_calls():
    import pytest

    router = FakeRouter([])
    with pytest.raises(ValueError, match="mode B"):
        investigate(_ctx(), {"training_labels": {}}, router, lambda *_: None)
    assert router.steps == []


def test_invalid_mode_b_labels_fail_before_model_calls():
    import pytest

    ctx = _ctx()
    ctx.mode = "B"
    router = FakeRouter([])
    with pytest.raises(ValueError):
        investigate(
            ctx,
            {"training_labels": {"reference": [], "observed": []}},
            router,
            lambda *_: None,
        )
    assert router.steps == []


def test_sandbox_observations_reach_hypothesis_generation(monkeypatch):
    from core.agent import loop

    report = {
        "status": "completed",
        "observations": {"rows": 2},
        "causal_confirmation": False,
    }
    monkeypatch.setattr(loop, "generated_check", lambda *a, **kw: report)
    captured = []
    hypotheses = [_hyp("h1", "other", 0.4)]

    def propose(router, case, **kwargs):
        captured.append(case)
        return hypotheses

    monkeypatch.setattr(loop, "propose_hypotheses", propose)
    result = investigate(
        _ctx(),
        {},
        FakeRouter(hypotheses),
        lambda *_: None,
        probes=(),
        causal_checks=False,
        sandbox_check={"question": "Count rows", "data": {"rows": [1, 2]}},
    )
    assert captured[0]["sandbox_check"] == report
    assert result.steps[0]["kind"] == "sandbox_check"
    assert result.status == "inconclusive"


def test_public_demo_rejects_generated_checks_before_calls(monkeypatch):
    import pytest

    from core.config import get_settings

    monkeypatch.setattr(get_settings(), "public_demo", True)
    router = FakeRouter([])
    with pytest.raises(ValueError, match="public demo"):
        investigate(_ctx(), {}, router, lambda *_: None, sandbox_check={})
    assert router.steps == []


def test_declined_paid_pair_never_runs_and_is_persisted():
    router = FakeRouter([_hyp("h1", "lr_too_low", 0.9)])
    runner, calls = _runner({"h1": ([0.5, 0.5], [0.1, 0.1])})
    proposals = []

    def decline(proposal):
        proposals.append(proposal)
        return False

    result = investigate(
        _ctx(),
        {},
        router,
        runner,
        probes=[],
        explain=False,
        intervention_cost_usd=0.1,
        approve_paid_step=decline,
    )
    assert calls == []
    assert len(proposals) == 1
    assert proposals[0]["target"]["patch"] == {"learning_rate": 0.0001}
    assert result.steps[-1]["reason"] == "paid causal check declined"
    assert result.steps[-2]["approved"] is False


def test_pair_failure_keeps_reservation(tmp_path):
    import pytest

    from core.db.spending import SpendingLedger

    ledger = SpendingLedger(tmp_path / "budget.db", 1)
    router = FakeRouter([_hyp("h1", "lr_too_low", 0.9)])

    def failed_pair(target, control):
        assert ledger.snapshot()["committed_usd"] == 0.1
        raise TimeoutError("unknown job outcome")

    with pytest.raises(TimeoutError):
        investigate(
            _ctx(),
            {},
            router,
            failed_pair,
            probes=[],
            explain=False,
            intervention_cost_usd=0.1,
            spending=ledger,
        )
    assert ledger.snapshot()["operations"][0]["status"] == "pending"


def test_calibrated_priors_reach_planner_without_changing_default():
    router = FakeRouter([_hyp("h1", "lr_too_low", 0.9)])
    result = investigate(
        _ctx(),
        {},
        router,
        lambda *_: None,
        probes=[],
        causal_checks=False,
        prior_temperature=2,
    )
    assert result.steps[0]["kind"] == "prior_calibration"
    assert result.steps[1]["probs"]["h1"] == 0.75
    assert result.steps[1]["probs"]["unlisted"] == 0.25
