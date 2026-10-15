import pytest

from bench.faults.recipe import STRENGTHS, inject

HEALTHY = {"learning_rate": 1e-4, "n_epochs": 1, "lora_r": 16, "lora_alpha": 16}


def test_faults_change_only_their_field():
    assert inject(HEALTHY, "lr_too_low", 0.1)["learning_rate"] == pytest.approx(1e-5)
    assert inject(HEALTHY, "too_many_epochs", 6)["n_epochs"] == 6
    small = inject(HEALTHY, "lora_rank_too_small", 2)
    assert small["lora_r"] == 2
    assert small["lora_alpha"] == 16
    assert HEALTHY["lora_r"] == 16


def test_every_planned_strength_applies():
    for fault, levels in STRENGTHS.items():
        for strength in levels.values():
            assert inject(HEALTHY, fault, strength) != HEALTHY


def test_unknown_fault():
    with pytest.raises(ValueError):
        inject(HEALTHY, "truncation", 0.5)
