from copy import deepcopy

import pytest

from bench.cost_fit import fit_model
from bench.sensitivity import summarize_study
from core.planner.job_cost import duration_from_report, estimate_job


def cost_rows():
    rows = []
    for n, e in [(50, 1), (200, 1), (1000, 1), (200, 3), (500, 2)]:
        tokens = n * 50 * e
        rows.append(
            {
                "model": "test",
                "status": "succeeded",
                "trained_tokens": tokens,
                "epochs": e,
                "examples": n,
                "job_id": f"{n}-{e}",
                "seconds": {
                    "datasets": 40,
                    "training_started": 100,
                    "training_completed": 100 + 20 + tokens / 1000 * 3 + (e - 1) * 4,
                    "finished": 150 + 20 + tokens / 1000 * 3 + (e - 1) * 4,
                },
            }
        )
    return rows


def test_cost_fit_recovers_coefficients_and_holds_back_validation():
    rows = cost_rows()
    result = fit_model(rows)
    assert result["coefficients"] == pytest.approx([20, 3, 4])
    assert result["validation"][0]["absolute_relative_error"] == pytest.approx(
        0, abs=1e-10
    )
    rows[-1]["seconds"]["training_completed"] += 100
    changed = fit_model(rows)
    assert changed["coefficients"] == pytest.approx(result["coefficients"])
    assert changed["validation"][0]["absolute_relative_error"] > 0.1


def test_planner_uses_only_model_matched_validated_duration():
    rows = cost_rows()
    for row in rows:
        row["model"] = "Qwen/Qwen3-0.6B"
    report = fit_model(rows)
    duration = duration_from_report(report, rows[0]["model"])
    estimate = estimate_job(rows[0]["model"], 500, 50, 2, duration=duration)
    assert estimate.seconds == pytest.approx(40 + 60 + 20 + 150 + 4 + 50)
    with pytest.raises(ValueError, match="another model"):
        duration_from_report(report, "other")
    report["validation"][0]["absolute_relative_error"] = 0.31
    with pytest.raises(ValueError, match="validation"):
        duration_from_report(report, rows[0]["model"])
    report["validation"][0]["absolute_relative_error"] = 0
    report["coefficients"][0] = -1
    with pytest.raises(ValueError, match="negative"):
        duration_from_report(report, rows[0]["model"])


def test_cost_fit_refuses_missing_holdback_and_failed_jobs():
    with pytest.raises(ValueError):
        fit_model(cost_rows()[:-1])
    rows = cost_rows()
    rows[0]["status"] = "failed"
    with pytest.raises(ValueError):
        fit_model(rows)


def sensitivity_rows():
    return [
        {
            "study": "local",
            "seed": s,
            "fault": fault,
            "hyperparameters": {"learning_rate": lr, "n_epochs": 3},
            "exact_match": score,
        }
        for s in (0, 1, 2)
        for fault, lr, score in [(None, 1e-4, 0.5), ("lr_too_low", 1e-5, 0.2)]
    ]


def test_sensitivity_reports_paired_changes():
    report = summarize_study(sensitivity_rows())
    assert report["curves"][0]["mean_change_from_healthy"] == pytest.approx(-0.3)
    assert report["curves"][0]["seed_sd"] == 0


@pytest.mark.parametrize(
    "change", ["mixed", "duplicate", "confounded", "extra", "missing"]
)
def test_sensitivity_rejects_invalid_comparisons(change):
    rows = deepcopy(sensitivity_rows())
    if change == "mixed":
        rows[0]["study"] = "remote"
    elif change == "duplicate":
        rows.append(rows[1])
    elif change == "confounded":
        rows[1]["hyperparameters"]["n_epochs"] = 4
    elif change == "extra":
        rows[1]["hyperparameters"]["weight_decay"] = 0.1
    else:
        rows = rows[1:]
    with pytest.raises(ValueError):
        summarize_study(rows)


def test_detection_threshold_is_observed_not_a_guarantee():
    from bench.sensitivity import detection_report

    rows = sensitivity_rows()
    for row in rows:
        row.update(ranking=[("lr_too_low", 0.8)], cost_usd=0.001)
    report = detection_report(rows)
    assert report["curves"][0]["top1_recovery"] == 1
    threshold = report["thresholds"][0]
    assert threshold["minimum_tested_value_at_observed_80_percent"] == 1e-5
    assert (
        threshold["minimum_tested_value_with_95_percent_lower_bound_at_80_percent"]
        is None
    )
    assert report["healthy"][0]["recipe_top1_on_healthy"] == 3
    rows[1].update(ranking=[], error="structured output failed")
    assert detection_report(rows)["curves"][0]["top1_recovery"] == pytest.approx(2 / 3)


def test_detection_observation_excludes_truth_and_paths():
    from scripts.measure_detection_sensitivity import observation

    row = {
        "study": "tokenfactory",
        "fault": "too_many_epochs",
        "seed": 2,
        "run": "secret-fault-name",
        "hyperparameters": {"n_epochs": 18},
        "steps": [1, 2],
        "train_loss": [1, 0.5],
        "valid_loss": [1, 2],
        "exact_match": 0.6,
    }
    obs = observation(row, {"exact_match": 0.5})
    assert not {"fault", "seed", "run", "study", "source"}.intersection(obs)
    assert "secret-fault-name" not in str(obs)
    assert obs["job"]["hyperparameters"]["n_epochs"] == 18
