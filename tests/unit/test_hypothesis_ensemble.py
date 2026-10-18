import pytest

from core.agent.ensemble import combine, sample_hypotheses
from core.agent.loop import investigate
from core.agent.router import Router
from core.agent.schemas import HypothesisList
from core.connectors.token_factory import CallResult
from core.telemetry.calls import CallLog
from tests.unit.test_loop import FakeRouter, _ctx, _hyp


def test_mean_preserves_missing_mass_and_merges_duplicate_classes():
    draws = [
        [_hyp("a", "lr_too_low", 0.3), _hyp("b", "lr_too_low", 0.2)],
        [_hyp("c", "other", 0.6)],
    ]
    merged, support = combine(draws)
    assert [(h.id, h.fault_class, h.prior) for h in merged] == [
        ("h1", "other", 0.3),
        ("h2", "lr_too_low", 0.25),
    ]
    assert support == {"other": 1, "lr_too_low": 1}
    assert sum(h.prior for h in merged) == 0.55
    assert draws[0][0].id == "a"


class Provider:
    def __init__(self):
        self.messages = []

    def model(self, role):
        return "test-super"

    def structured(self, model, messages, schema):
        assert schema is HypothesisList
        self.messages.append(messages)
        return HypothesisList(hypotheses=[_hyp("h1", "lr_too_low", 0.8)]), [
            CallResult(model, "{}", 100, 20, 0.01, 10)
        ]


def test_loop_records_draws_and_charges_every_call():
    provider = Provider()
    router = Router(provider, CallLog("ensemble"))
    result = investigate(
        _ctx(),
        {},
        router,
        lambda *args: pytest.fail("No training"),
        hypothesis_samples=3,
        causal_checks=False,
    )
    assert len(provider.messages) == 3
    assert all(m == provider.messages[0] for m in provider.messages)
    assert result.llm_cost_usd == pytest.approx(0.03)
    assert result.steps[0]["kind"] == "hypothesis_sampling"
    assert result.steps[0]["support_counts"] == {"lr_too_low": 3}
    assert len(result.steps[0]["draws"]) == 3
    assert result.steps[1]["kind"] == "hypotheses"
    assert result.status == "inconclusive"


def test_budget_stops_before_the_next_call():
    provider = Provider()
    router = Router(provider, CallLog("ensemble"))
    with pytest.raises(ValueError, match="budget"):
        sample_hypotheses(router, {}, mode="A", samples=3, budget_usd=0.01)
    assert len(provider.messages) == 1
    assert router.log.cost_usd == 0.01


@pytest.mark.parametrize("count", [0, 6, True, 1.5])
def test_invalid_sample_count(count):
    with pytest.raises(ValueError, match="hypothesis_samples"):
        investigate(
            _ctx(), {}, FakeRouter([]), lambda *args: None, hypothesis_samples=count
        )


def test_frozen_hypotheses_cannot_trigger_hidden_resampling():
    with pytest.raises(ValueError, match="Frozen"):
        investigate(
            _ctx(),
            {},
            FakeRouter([]),
            lambda *args: None,
            initial_hypotheses=[_hyp("h1", "other", 0.5)],
            hypothesis_samples=3,
        )


def test_settings_validate_live_sample_count():
    from pydantic import ValidationError

    from core.config import Settings

    assert Settings(_env_file=None, hypothesis_samples=3).hypothesis_samples == 3
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hypothesis_samples=6)


def test_sampling_filters_modes_and_unverified_sources():
    class DirtyProvider(Provider):
        def structured(self, model, messages, schema):
            valid = _hyp("a", "lr_too_low", 0.7)
            valid.sources = [
                "https://trusted.example/issue",
                "https://fake.example/issue",
            ]
            invalid = _hyp("b", "double_label_shift", 0.3)
            return HypothesisList(hypotheses=[valid, invalid]), []

    router = Router(DirtyProvider(), CallLog("ensemble"))
    hypotheses, record = sample_hypotheses(
        router,
        {"known_issues": [{"url": "https://trusted.example/issue"}]},
        mode="A",
        samples=2,
        budget_usd=1,
    )
    assert len(hypotheses) == 1
    assert hypotheses[0].sources == ["https://trusted.example/issue"]
    assert record["support_counts"] == {"lr_too_low": 2}
