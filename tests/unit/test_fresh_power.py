import pytest

from scripts.fresh_answer_power import decide

RULE = {
    "candidate_seeds": [3, 4],
    "anticipated_effect": 0.16,
    "min_effect": 0.08,
    "trials": 1000,
    "random_seed": 3101,
    "minimum_lower_mc_bound": 0.8,
}
DIGESTS = {"healthy": ["a", "b", "c"], "damaged": ["d", "e", "f"]}


def test_stop_on_duplicate_weights_or_zero_sd():
    groups = {"healthy": [0.8] * 3, "damaged": [0.3] * 3}
    assert decide(groups, DIGESTS, RULE)["reason"] == "zero_sd"
    duplicate = {**DIGESTS, "healthy": ["a"] * 3}
    assert decide(groups, duplicate, RULE)["reason"] == "identical_weights"


def test_first_passing_candidate_is_locked(monkeypatch):
    seen = []

    def simulate(**kwargs):
        seen.append(kwargs)
        return {"mc_interval_95": [0.9, 0.99]}

    monkeypatch.setattr("scripts.fresh_answer_power.simulate", simulate)
    groups = {"healthy": [0.8, 0.81, 0.79], "damaged": [0.3, 0.31, 0.29]}
    result = decide(groups, DIGESTS, RULE)
    assert result["selected_seeds"] == [2101, 2102, 2103]
    assert len(seen) == 1
    assert seen[0]["random_seed"] == 3101
    assert seen[0]["noise_sd"] == pytest.approx(0.01)


def test_stop_without_inventing_extra_seeds(monkeypatch):
    monkeypatch.setattr(
        "scripts.fresh_answer_power.simulate",
        lambda **kwargs: {"mc_interval_95": [0.4, 0.6]},
    )
    result = decide(
        {"healthy": [0.8, 0.81, 0.79], "damaged": [0.3, 0.31, 0.29]}, DIGESTS, RULE
    )
    assert result["reason"] == "insufficient_power"
    assert result["selected_seeds"] == []
    assert len(result["power"]) == 2
