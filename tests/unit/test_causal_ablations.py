import pytest

from bench.ablations import ReplayRouter
from bench.diagnosis import from_investigation
from core.agent.loop import investigate
from core.agent.schemas import HypothesisList
from core.planner.causal_belief import update_causal
from tests.unit.test_loop import FakeRouter, _ctx, _hyp, _runner


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("confirmed", 0.75),
        ("rejected", 0.25),
        ("inconclusive", 0.5),
    ],
)
def test_causal_outcome_updates_odds(status, expected):
    prior = {"h1": 0.5, "h2": 0.3, "unlisted": 0.2}
    classes = {"h1": "lr_too_low", "h2": "other", "unlisted": "unlisted"}
    posterior = update_causal(prior, classes, "h1", status)
    assert posterior["h1"] == pytest.approx(expected)
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior["h2"] / posterior["unlisted"] == pytest.approx(1.5)
    assert prior["h1"] == 0.5


def test_hypotheses_of_the_same_class_receive_the_same_evidence():
    prior = {"h1": 0.2, "h2": 0.3, "h3": 0.5}
    classes = {"h1": "lr_too_low", "h2": "lr_too_low", "h3": "other"}
    posterior = update_causal(prior, classes, "h1", "confirmed")
    assert posterior["h1"] + posterior["h2"] == pytest.approx(0.75)
    assert posterior["h1"] / posterior["h2"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("metrics", "status"),
    [
        (([0.50, 0.50], [0.10, 0.10]), "confirmed"),
        (([0.10, 0.10], [0.50, 0.50]), "rejected"),
        (([0.13, 0.13], [0.10, 0.10]), "inconclusive"),
    ],
)
def test_benchmark_reads_the_saved_posterior_once(metrics, status):
    hypotheses = [_hyp("h1", "lr_too_low", 0.6), _hyp("h2", "other", 0.4)]
    run_pair, _ = _runner({"h1": metrics})
    result = investigate(
        _ctx(),
        {},
        FakeRouter(hypotheses),
        run_pair,
        probes=(),
        explain=False,
        initial_hypotheses=hypotheses,
        min_effect=0.06,
        noise_sd=0.019,
        max_interventions=1,
    )
    step = next(s for s in result.steps if s["kind"] == "intervention")
    assert step["result"]["status"] == status
    posterior = step["probs"]["h1"]
    assert 0 < posterior < 1
    diagnosis = from_investigation(result)
    assert dict(diagnosis.ranking)["lr_too_low"] == pytest.approx(posterior)
    if status == "inconclusive":
        assert posterior == pytest.approx(step["prior_probs"]["h1"])


def run_order(policy, seed):
    hypotheses = [
        _hyp("h1", "too_many_epochs", 0.2),
        _hyp("h2", "lr_too_low", 0.5),
        _hyp("h3", "lora_rank_too_small", 0.3),
    ]
    run_pair, calls = _runner({h.id: ([0.1, 0.1], [0.5, 0.5]) for h in hypotheses})
    investigate(
        _ctx(),
        {},
        FakeRouter(hypotheses),
        run_pair,
        probes=(),
        initial_hypotheses=hypotheses,
        explain=False,
        policy=policy,
        random_seed=seed,
        max_interventions=2,
    )
    return calls


def test_fixed_order_and_intervention_limit():
    calls = run_order("fixed", 0)
    assert [list(patch) for patch in calls] == [
        ["n_epochs"],
        ["learning_rate"],
    ]


def test_random_order_is_repeatable_and_varies_between_seeds():
    orders = []
    for seed in range(10):
        first = run_order("random", seed)
        assert first == run_order("random", seed)
        assert len(first) == 2
        orders.append(repr(first))
    assert len(set(orders)) > 1


def test_replay_router_forbids_hypothesis_generation():
    router = ReplayRouter("offline", [])
    with pytest.raises(AssertionError, match="Unexpected LLM request"):
        router.structured("hypotheses", [], HypothesisList, prompt_version="test")
