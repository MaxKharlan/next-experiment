import pytest

from bench.faults.recipe import HEALTHY_HP, inject
from core.agent.context import InvestigationContext
from core.agent.schemas import Hypothesis
from core.connectors.token_factory import hyperparameter_problem
from core.interventions.pairs import make_pair, take_fraction


def _ctx(hp):
    return InvestigationContext(
        investigation_id="inv-1",
        mode="A",
        job_id="ftjob-1",
        job_config={"hyperparameters": hp},
    )


def _hyp(fault_class):
    return Hypothesis(
        id="h1", fault_class=fault_class, location="config", prior=0.5, rationale=""
    )


def _changed(arm, base):
    return {k for k, v in arm.hyperparameters(base).items() if base.get(k) != v}


@pytest.mark.parametrize(
    ("fault", "strength"),
    [("lr_too_low", 0.1), ("too_many_epochs", 3), ("lora_rank_too_small", 2)],
)
def test_control_costs_at_least_the_target(fault, strength):
    failed = inject(HEALTHY_HP, fault, strength)
    target, control = make_pair(_hyp(fault), _ctx(failed))
    assert (target.kind, control.kind) == ("target", "control")
    assert control.relative_cost(failed) >= target.relative_cost(failed)
    if fault != "too_many_epochs":
        assert control.relative_cost(failed) == target.relative_cost(failed)


def test_low_lr_target_restores_the_rate():
    failed = inject(HEALTHY_HP, "lr_too_low", 0.1)
    target, control = make_pair(_hyp("lr_too_low"), _ctx(failed))
    assert target.hyperparameters(failed)["learning_rate"] == pytest.approx(1e-4)
    assert _changed(target, failed) == {"learning_rate"}
    assert _changed(control, failed) == set()


def test_epochs_control_is_the_failed_recipe():
    failed = inject(HEALTHY_HP, "too_many_epochs", 6)
    target, control = make_pair(_hyp("too_many_epochs"), _ctx(failed))
    assert target.hyperparameters(failed)["n_epochs"] == 2
    assert control.hyperparameters(failed) == failed
    assert control.data_fraction == 1.0


def test_rank_target_keeps_the_lora_scale():
    # Token Factory's defaults: r 8, alpha 8
    failed = {**HEALTHY_HP, "lora_r": 8, "lora_alpha": 8}
    target, control = make_pair(_hyp("lora_rank_too_small"), _ctx(failed))
    t = target.hyperparameters(failed)
    assert (t["lora_r"], t["lora_alpha"]) == (32, 32)
    assert control.hyperparameters(failed) == failed


@pytest.mark.parametrize(
    "recipe",
    [
        HEALTHY_HP,
        {
            **HEALTHY_HP,
            "learning_rate": 1e-5,
            "n_epochs": 3,
            "lora_r": 8,
            "lora_alpha": 8,
        },
        {**HEALTHY_HP, "learning_rate": 2e-5, "n_epochs": 18},
    ],
)
@pytest.mark.parametrize(
    "fault", ["lr_too_low", "too_many_epochs", "lora_rank_too_small"]
)
def test_every_arm_stays_inside_token_factory_ranges(recipe, fault):
    try:
        arms = make_pair(_hyp(fault), _ctx(recipe))
    except ValueError:  # e.g. one epoch cannot be too many
        return
    for arm in arms:
        assert hyperparameter_problem(arm.hyperparameters(recipe)) is None


def test_impossible_pairs_raise():
    with pytest.raises(ValueError):
        make_pair(_hyp("too_many_epochs"), _ctx(HEALTHY_HP))
    with pytest.raises(NotImplementedError):
        make_pair(_hyp("truncation"), _ctx(HEALTHY_HP))


def test_take_fraction_is_stable():
    rows = list(range(10))
    assert take_fraction(rows, 0.3) == take_fraction(rows, 0.3)
    assert len(take_fraction(rows, 0.3)) == 3
    assert take_fraction(rows, 1.0) == rows
