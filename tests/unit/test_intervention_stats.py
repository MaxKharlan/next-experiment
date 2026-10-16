import pytest

from core.interventions.stats import (
    causal_check,
    min_detectable_effect,
    noise_floor,
    permutation_p,
    seeds_for_power,
)

# Exact match of Qwen3-0.6B on 100 BANKING77 examples, 3 seeds per config
# (local B9 sweep, 500 examples, 3 epochs; docs/b10-noise-floor.md)
EXACT_MATCH = {
    "healthy": [0.43, 0.41, 0.42],
    "lr_too_low-mild": [0.05, 0.05, 0.05],
    "lr_too_low-strong": [0.03, 0.03, 0.03],
    "lora_rank_too_small-mild": [0.44, 0.41, 0.40],
    "lora_rank_too_small-strong": [0.43, 0.39, 0.43],
}
VALID_LOSS = {
    "healthy": [0.53198, 0.51790, 0.51020],
    "lr_too_low-mild": [1.97426, 1.96338, 1.96114],
    "lora_rank_too_small-mild": [0.53382, 0.51787, 0.52726],
}


def test_noise_floor_pools_configs():
    floor = noise_floor(EXACT_MATCH)
    assert floor.sd == pytest.approx(0.0146, abs=1e-4)
    assert (floor.runs, floor.groups) == (15, 5)


def test_noise_floor_needs_two_seeds():
    with pytest.raises(ValueError):
        noise_floor({"healthy": [0.4]})


def test_power():
    assert seeds_for_power(0.37, 0.0146) == 2
    assert seeds_for_power(0.02, 0.0146) == 7
    assert min_detectable_effect(0.0146, 3) == pytest.approx(0.0297, abs=1e-4)


def test_permutation_p_for_full_separation():
    assert permutation_p([3, 4, 5], [0, 1, 2]) == pytest.approx(1 / 20)
    assert permutation_p([0, 1, 2], [3, 4, 5]) == 1.0


def test_low_lr_is_confirmed():
    sd = noise_floor(EXACT_MATCH).sd
    r = causal_check(
        EXACT_MATCH["healthy"],
        EXACT_MATCH["lr_too_low-mild"],
        min_effect=2 * sd,
        noise_sd=sd,
    )
    assert r.status == "confirmed"
    assert r.effect == pytest.approx(0.37)


def test_small_rank_is_rejected():
    sd = noise_floor(EXACT_MATCH).sd
    r = causal_check(
        EXACT_MATCH["healthy"],
        EXACT_MATCH["lora_rank_too_small-mild"],
        min_effect=2 * sd,
        noise_sd=sd,
    )
    assert r.status == "rejected"


def test_loss_lower_is_better():
    r = causal_check(
        VALID_LOSS["healthy"],
        VALID_LOSS["lr_too_low-mild"],
        min_effect=0.05,
        higher_is_better=False,
    )
    assert r.status == "confirmed"
    assert r.effect > 1.4


def test_noisy_arms_are_inconclusive():
    r = causal_check([0.50, 0.30, 0.45], [0.35, 0.40, 0.20], min_effect=0.05)
    assert r.status == "inconclusive"
    assert r.ci_low < 0.05 < r.ci_high


def test_needs_two_seeds_per_arm():
    with pytest.raises(ValueError):
        causal_check([0.4], [0.1, 0.2], min_effect=0.05)


def test_two_seeds_rely_on_the_noise_floor():
    sd = noise_floor(EXACT_MATCH).sd
    r = causal_check([0.43, 0.41], [0.05, 0.05], min_effect=2 * sd, noise_sd=sd)
    assert r.p_value > 0.05
    assert r.status == "confirmed"
