import math

import pytest

from scripts.measure_model_costs import build_plan


def test_budget_plan_has_full_grid_and_validation_holdback():
    rows = ['{"messages":[]}'] * 1000
    plan = build_plan(rows, b"validation", 9)
    assert len(plan["runs"]) == 15
    assert len({p["run"] for p in plan["runs"]}) == 15
    assert plan["reserved_usd"] < 9
    assert {(p["examples"], p["epochs"]) for p in plan["runs"]} == {
        (50, 1),
        (200, 1),
        (1000, 1),
        (200, 3),
        (500, 2),
    }
    assert (
        build_plan(rows, b"changed", 9)["validation_sha256"]
        != plan["validation_sha256"]
    )


@pytest.mark.parametrize("budget", [0, -1, math.nan, math.inf, 11, 0.01])
def test_budget_rejects_invalid_or_insufficient(budget):
    with pytest.raises(ValueError):
        build_plan(["{}"] * 1000, b"valid", budget)
