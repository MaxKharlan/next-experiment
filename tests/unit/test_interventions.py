import pytest

from bench.faults.recipe import HEALTHY_HP, inject
from core.agent.context import InvestigationContext
from core.agent.schemas import Hypothesis
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


def test_rank_control_matches_the_lora_scale():
    failed = inject(HEALTHY_HP, "lora_rank_too_small", 2)
    target, control = make_pair(_hyp("lora_rank_too_small"), _ctx(failed))
    t, c = target.hyperparameters(failed), control.hyperparameters(failed)
    assert (t["lora_r"], t["lora_alpha"]) == (16, 16)
    assert (c["lora_r"], c["lora_alpha"]) == (2, 2)


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
