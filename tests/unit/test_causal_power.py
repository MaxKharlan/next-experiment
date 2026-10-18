import pytest

from bench.causal_power import simulate
from core.interventions import stats


def run(**kwargs):
    return simulate(
        **{
            "effect": 0.1,
            "noise_sd": 0.019,
            "seeds": 3,
            "trials": 50,
            "min_effect": 0.06,
            **kwargs,
        }
    )


def test_reproducible_and_counts_cover_all_trials():
    result = run()
    assert result == run()
    assert sum(result["counts"].values()) == 50
    low, high = result["mc_interval_95"]
    assert 0 <= low <= result["confirmation_rate"] <= high <= 1
    assert result["permutation_gate_active"] is True
    assert run(seeds=2)["permutation_gate_active"] is False


def test_no_noise_extremes_and_permutation_minimum():
    assert run(noise_sd=0, effect=0)["counts"]["rejected"] == 50
    assert run(noise_sd=0, effect=0.1)["confirmation_rate"] == 1
    assert stats.permutation_p([1, 1], [0, 0]) == pytest.approx(1 / 6)
    assert stats.permutation_p([1] * 3, [0] * 3) == pytest.approx(1 / 20)
    assert stats.permutation_p([1] * 3, [1] * 3) == 1


def test_large_test_samples_without_enumerating(monkeypatch):
    def forbidden(*args):
        pytest.fail("Large permutation spaces must not be enumerated")

    monkeypatch.setattr(stats, "combinations", forbidden)
    p = stats.permutation_p([1] * 20, [0] * 20)
    assert p == pytest.approx(1 / 20001)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_sd": -1},
        {"effect": float("nan")},
        {"min_effect": float("inf")},
        {"seeds": 1},
        {"seeds": True},
        {"trials": 0},
    ],
)
def test_invalid_simulation_inputs(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)


@pytest.mark.parametrize(
    "fix,control", [([], [1]), ([float("nan")], [1]), ([1], [float("inf")])]
)
def test_invalid_permutation_inputs(fix, control):
    with pytest.raises(ValueError):
        stats.permutation_p(fix, control)
