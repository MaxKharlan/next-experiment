import json

import pytest

from bench.sweep_analysis import (
    aggregate_outcomes,
    recipe_name,
    strength_tier,
    strength_transfer,
    validate_matrix,
)


def row(recipe, seed, value):
    label = "healthy" if recipe == "healthy" else recipe.rsplit("-", 1)[0]
    return {
        "id": f"{recipe}-s{seed}",
        "label": label,
        "seed": seed,
        "features": [value] * 5,
    }


def miniature_rows():
    recipes = (
        "healthy",
        "lr_too_low-mild",
        "lr_too_low-strong",
        "too_many_epochs-mild",
        "too_many_epochs-strong",
        "lora_rank_too_small-mild",
        "lora_rank_too_small-strong",
    )
    values = {
        "healthy": 0,
        "lr_too_low-mild": 10,
        "lr_too_low-strong": 10,
        "too_many_epochs-mild": 20,
        "too_many_epochs-strong": 20,
        "lora_rank_too_small-mild": 30,
        "lora_rank_too_small-strong": 30,
    }
    return [
        row(recipe, seed, values[recipe]) for seed in range(4) for recipe in recipes
    ]


def test_recipe_and_strength_are_derived_from_validated_id():
    assert recipe_name(row("lr_too_low-mild", 3, 0)) == "lr_too_low-mild"
    assert strength_tier(row("lr_too_low-mild", 3, 0)) == "mild"
    assert strength_tier(row("healthy", 3, 0)) == "healthy"
    broken = row("healthy", 3, 0)
    broken["id"] = "healthy-s4"
    with pytest.raises(ValueError, match="does not end"):
        recipe_name(broken)


def test_matrix_rejects_missing_and_unexpected_runs():
    rows = miniature_rows()
    validate_matrix(rows, tuple(range(4)))
    with pytest.raises(ValueError, match="Incomplete sweep"):
        validate_matrix(rows[:-1], tuple(range(4)))


def test_cross_strength_uses_disjoint_seeds_and_opposite_strengths():
    report = strength_transfer(miniature_rows(), split_seed=2)
    assert report["accuracy"] == 1
    assert report["balanced_accuracy"] == 1
    assert len(report["folds"]) == 4
    for fold in report["folds"]:
        assert set(fold["train_ids"]).isdisjoint(fold["test_ids"])
        assert len(fold["train_ids"]) == len(fold["test_ids"]) == 8


def test_outcomes_compare_each_recipe_to_the_same_seed(tmp_path):
    for item in miniature_rows():
        recipe = recipe_name(item)
        exact_match = 0.5 if recipe == "healthy" else item["seed"] * 0.1
        run = tmp_path / item["id"]
        run.mkdir()
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "seed": item["seed"],
                    "final": {
                        "exact_match": exact_match,
                        "valid_loss": 1.0,
                        "valid_mean_token_accuracy": 0.5,
                    },
                    "wall_seconds": 2.0,
                }
            )
        )
    report = aggregate_outcomes(tmp_path, expected_per_recipe=4, bootstrap_samples=100)
    healthy = report["recipes"]["healthy"]
    assert healthy["exact_match"]["mean"] == 0.5
    comparison = report["recipes"]["lr_too_low-mild"]["paired_exact_match_vs_healthy"]
    assert comparison["mean_difference"] == pytest.approx(-0.35)
    assert (comparison["wins"], comparison["ties"], comparison["losses"]) == (
        0,
        0,
        4,
    )
