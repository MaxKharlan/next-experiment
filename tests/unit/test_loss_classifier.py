import copy

import pytest

from bench.loss_classifier import curve_features, evaluate, fit, predict


def test_features_on_linear_curves_ignore_recipe_and_outcome_metrics():
    records = [
        {
            "step": i,
            "train_loss": 3 - i,
            "valid_loss": 4 - i,
            "exact_match": 999,
            "lr": 100,
        }
        for i in range(3)
    ]
    assert curve_features(records) == pytest.approx([-2, -2, 1, 1, -1])
    sparse = [
        {"step": 0, "valid_loss": 4},
        {"step": 1, "train_loss": 3},
        {"step": 2, "train_loss": 2},
        {"step": 2, "valid_loss": 2},
    ]
    assert curve_features(sparse)[:3] == pytest.approx([-1, -2, 0])


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{"step": 0, "train_loss": 1}],
        [{"step": 0, "train_loss": 1, "valid_loss": 1}] * 2,
        [
            {"step": 0, "train_loss": 1, "valid_loss": 1},
            {"step": 1, "train_loss": float("nan"), "valid_loss": 1},
        ],
    ],
)
def test_bad_curves_rejected(records):
    with pytest.raises(ValueError):
        curve_features(records)


def rows():
    return [
        {"id": f"{label}-{seed}", "label": label, "seed": seed, "features": [value] * 5}
        for seed in [0, 1, 2]
        for label, value in [("a", 0.0), ("b", 10.0)]
    ]


def test_grouped_folds_fit_scaling_without_test_seed():
    data = rows()
    data[0]["features"] = [1000] * 5
    result = evaluate(data)
    fold = next(f for f in result["folds"] if f["excluded_seed"] == 0)
    assert fold["model"]["mean"] == [5] * 5
    assert set(fold["train_ids"]).isdisjoint(fold["test_ids"])
    assert len(result["predictions"]) == len(data)


def test_separable_classes_and_zero_variance_features():
    data = rows()
    for row in data:
        row["features"][-1] = 7
    original = copy.deepcopy(data)
    assert evaluate(data)["accuracy"] == 1
    model = fit(data)
    assert model["scale"][-1] == 1
    assert predict(model, data[0]["features"]) == "a"
    assert data == original


def test_missing_class_in_training_fold_rejected():
    with pytest.raises(ValueError, match="all recipe classes"):
        evaluate([rows()[0], rows()[3]])
