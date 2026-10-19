import math

import pytest

from bench.prior_calibration import audit, fit, metrics
from core.agent.prior_calibration import calibrate, distribution, scale_distribution
from core.agent.schemas import Hypothesis


def hyp(name, p):
    return Hypothesis(
        id=name, fault_class="lr_too_low", location=name, prior=p, rationale=""
    )


def test_scaling_preserves_mass_duplicate_locations_and_identity():
    hypotheses = [hyp("a", 0.3), hyp("b", 0.1)]
    assert distribution(hypotheses) == {"lr_too_low": 0.4, "unlisted": 0.6}
    identity, _ = calibrate(hypotheses, 1)
    assert [h.prior for h in identity] == pytest.approx([0.3, 0.1])
    adjusted, step = calibrate(hypotheses, 2)
    assert adjusted[0].prior / adjusted[1].prior == pytest.approx(3)
    assert sum(h.prior for h in adjusted) + step["after"]["unlisted"] == pytest.approx(
        1
    )
    assert hypotheses[0].prior == 0.3


@pytest.mark.parametrize("t", [0, -1, math.nan, math.inf, True])
def test_bad_temperatures(t):
    with pytest.raises(ValueError):
        scale_distribution({"x": 0.5, "unlisted": 0.5}, t)


def test_fit_uses_whole_distribution_and_case_exclusion():
    rows = [
        {
            "case": str(i),
            "probabilities": {"x": 0.9, "unlisted": 0.1},
            "target": "x" if i < 2 else "unlisted",
        }
        for i in range(4)
    ]
    t = fit(rows)
    assert metrics(rows, t)["nll"] < metrics(rows, 1)["nll"]
    report = audit(rows)
    for fold in report["leave_one_case_out"]:
        assert fold["case"] not in fold["training_cases"]
        assert len(fold["training_cases"]) == 3
