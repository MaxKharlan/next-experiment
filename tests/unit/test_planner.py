import pytest

from core.agent.context import InvestigationContext
from core.planner.eig import best_probe, entropy, expected_gain, update
from core.planner.outcome_model import likelihood, load_signatures, outcomes
from core.planner.stopping import next_action
from core.probes.level0.recipe import LEVEL0

CLASSES = {"h1": "lr_too_low", "h2": "chat_template_mismatch", "unlisted": "unlisted"}
PRIORS = {"h1": 0.4, "h2": 0.4, "unlisted": 0.2}


def _ctx(**job):
    return InvestigationContext("inv-1", "A", "ftjob-1", job)


def test_signatures_cover_every_probe_outcome():
    table = load_signatures()
    for probe in LEVEL0:
        assert probe.name in table
        assert set(outcomes(probe.name)) == set(probe.outcomes)
    assert likelihood("lr_range", "lr_too_low", "low") == 0.95
    assert likelihood("lr_range", "truncation", "low") == 0.30


def test_bayes_update_moves_mass_to_the_matching_class():
    after = update(PRIORS, CLASSES, "lr_range", "low")
    assert sum(after.values()) == pytest.approx(1)
    assert after["h1"] > 0.6 > PRIORS["h1"]


def test_informative_probe_is_chosen_first():
    assert entropy({"a": 0.5, "b": 0.5}) == pytest.approx(1.0)
    assert expected_gain(PRIORS, CLASSES, "lr_range") > expected_gain(
        PRIORS, CLASSES, "lora_rank"
    )
    probe, score = best_probe(PRIORS, CLASSES, {"lr_range": 0.0, "lora_rank": 0.0})
    assert probe == "lr_range"
    assert score > 0
    assert best_probe(PRIORS, CLASSES, {}) == (None, 0.0)


def test_next_action():
    kw = {"gain_bits": 0.2, "budget_left_usd": 1.0, "intervention_cost_usd": 0.0}
    assert next_action(PRIORS, threshold=0.6, testable={"h1"}, **kw)[0] == "probe"
    high = {"h1": 0.7, "h2": 0.2, "unlisted": 0.1}
    assert next_action(high, threshold=0.6, testable={"h1"}, **kw)[0] == "intervene"
    assert next_action(high, threshold=0.6, testable=set(), **kw)[0] == "stop"
    poor = dict(kw, budget_left_usd=0.01, intervention_cost_usd=0.5)
    assert next_action(high, threshold=0.6, testable={"h1"}, **poor)[0] == "stop"


@pytest.mark.parametrize(
    ("hp", "probe", "outcome"),
    [
        ({"learning_rate": 1e-5}, "lr_range", "low"),
        ({"learning_rate": 1e-4}, "lr_range", "ok"),
        ({"n_epochs": 6}, "epoch_count", "high"),
        ({"lora_r": 2}, "lora_rank", "low"),
    ],
)
def test_recipe_probes(hp, probe, outcome):
    run = next(p for p in LEVEL0 if p.name == probe).run(_ctx(hyperparameters=hp))
    assert run.outcome == outcome
    assert run.raw_ref == f"investigations/inv-1/probes/{probe}.json"


@pytest.mark.parametrize(
    ("train", "valid", "outcome"),
    [
        ([3.25, 2.66, 2.46], [2.79, 2.51, 2.45], "barely_moved"),
        ([1.17, 0.33, 0.08], [0.45, 1.03, 2.03], "overfit"),
        ([1.67, 0.57, 0.26], [0.97, 0.56, 0.40], "normal"),
    ],
)
def test_loss_curve(train, valid, outcome):
    points = [{"train_loss": t, "valid_loss": v} for t, v in zip(train, valid)]
    run = next(p for p in LEVEL0 if p.name == "loss_curve").run(
        _ctx(checkpoints=points)
    )
    assert run.outcome == outcome
