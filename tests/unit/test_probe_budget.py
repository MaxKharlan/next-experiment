import pytest

from core.agent.loop import investigate
from core.probes.base import ProbeResult
from core.probes.level0.recipe import LearningRateRange
from tests.unit.test_loop import FakeRouter, _ctx, _hyp, _runner


class PaidProbe(LearningRateRange):
    def __init__(self, estimate, actual):
        self.estimate = estimate
        self.actual = actual
        self.calls = 0

    def estimate_cost(self, ctx):
        return self.estimate

    def run(self, ctx):
        self.calls += 1
        return ProbeResult(
            outcome="low", raw_ref="probe.json", summary="{}", cost_usd=self.actual
        )


@pytest.mark.parametrize("policy", ["fixed", "random", "eig"])
def test_unaffordable_probe_does_not_run(policy):
    probe = PaidProbe(2, 2)
    router = FakeRouter([_hyp("h1", "lr_too_low", 0.4)])
    runner, calls = _runner({})
    result = investigate(
        _ctx(),
        {},
        router,
        runner,
        probes=[probe],
        policy=policy,
        threshold=1,
        explain=False,
    )
    assert probe.calls == 0
    assert calls == []
    assert result.steps[-1]["reason"] == "the budget does not cover this probe"
    assert result.steps[-1]["estimated_cost_usd"] == 2


def test_actual_probe_cost_blocks_next_causal_check():
    probe = PaidProbe(0.5, 0.8)
    router = FakeRouter([_hyp("h1", "lr_too_low", 0.4)])
    runner, calls = _runner({})
    ctx = _ctx()
    result = investigate(
        ctx,
        {},
        router,
        runner,
        probes=[probe],
        policy="fixed",
        intervention_cost_usd=0.3,
        explain=False,
    )
    assert probe.calls == 1
    assert calls == []
    assert ctx.budget_left_usd == pytest.approx(0.2)
    step = next(s for s in result.steps if s["kind"] == "probe")
    assert step["cost_usd"] == 0.8
    assert step["estimated_cost_usd"] == 0.5


def test_probe_overrun_does_not_trigger_paid_explanation():
    probe = PaidProbe(1, 1.1)
    router = FakeRouter([_hyp("h1", "lr_too_low", 0.4)])
    runner, calls = _runner({})
    result = investigate(
        _ctx(),
        {},
        router,
        runner,
        probes=[probe],
        policy="fixed",
        intervention_cost_usd=0.1,
    )
    assert probe.calls == 1
    assert calls == []
    assert "interpret_probe" not in router.steps
    assert result.status == "inconclusive"


@pytest.mark.parametrize("estimate", [-1, float("nan"), float("inf")])
def test_invalid_estimate_fails_before_execution(estimate):
    probe = PaidProbe(estimate, 0)
    router = FakeRouter([_hyp("h1", "lr_too_low", 0.4)])
    runner, _ = _runner({})
    with pytest.raises(ValueError, match="probe cost estimate"):
        investigate(_ctx(), {}, router, runner, probes=[probe], policy="fixed")
    assert probe.calls == 0
