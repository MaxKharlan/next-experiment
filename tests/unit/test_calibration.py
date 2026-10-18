from types import SimpleNamespace

import pytest

from bench.calibration import analyze, reliability, selected_cause


def test_brier_and_ece_match_hand_calculation():
    result = reliability([(0.8, True), (0.6, False), (0.2, False), (1.0, True)], bins=2)
    assert result["binary_brier"] == pytest.approx(0.11)
    assert result["ece"] == pytest.approx(0.15)
    assert [b["count"] for b in result["bins"]] == [1, 3]
    assert result["bins"][1]["accuracy"] == pytest.approx(2 / 3)


def test_endpoints_and_empty_bins():
    result = reliability([(0, False), (1, True)], bins=5)
    assert result["binary_brier"] == result["ece"] == 0
    assert result["bins"][1]["mean_confidence"] is None
    assert result["bins"][-1]["count"] == 1


def test_selection_uses_probability_without_renormalizing():
    assert selected_cause([["promoted", 0.1], ["leading", 0.6]]) == ["leading", 0.6]
    assert selected_cause([["z", 0.4], ["a", 0.4]]) == ["a", 0.4]


@pytest.mark.parametrize(
    "ranking",
    [
        [],
        [["a", float("nan")]],
        [["a", float("inf")]],
        [["a", -0.1]],
        [["a", True]],
        [["a", 0.7], ["b", 0.6]],
        [["a", 0.2], ["a", 0.3]],
    ],
)
def test_invalid_distributions_are_rejected(ranking):
    with pytest.raises(ValueError):
        selected_cause(ranking)


@pytest.fixture
def cases():
    return [
        SimpleNamespace(id="fault", truth=["lr_too_low"], healthy=False),
        SimpleNamespace(id="healthy", truth=[], healthy=True),
    ]


@pytest.fixture
def records():
    return [
        {"case": "fault", "variant": "mixed", "ranking": [["lr_too_low", 0.8]]},
        {"case": "healthy", "variant": "mixed", "ranking": [["other", 0.2]]},
    ]


def test_healthy_is_not_inferred_from_residual_mass(cases, records):
    output = analyze(records, cases)["methods"]["mixed"]
    assert output["metrics"]["all"]["binary_brier"] == pytest.approx(0.04)
    assert output["metrics"]["healthy"]["accuracy"] == 0
    assert output["metrics"]["faults"]["accuracy"] == 1
    assert output["cases"][1]["correct"] is False


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong-split"])
def test_case_coverage_is_enforced(cases, records, problem):
    if problem == "missing":
        records.pop()
    elif problem == "duplicate":
        records.append(records[0])
    else:
        records[0]["case"] = "heldout-case"
    with pytest.raises(ValueError):
        analyze(records, cases)


@pytest.mark.parametrize("bins", [0, 21, True, 1.5])
def test_invalid_bin_count(bins):
    with pytest.raises(ValueError):
        reliability([(0.5, True)], bins)


def test_cli_records_provenance_and_refuses_overwrite(
    tmp_path, monkeypatch, cases, records
):
    import json
    import sys

    from bench import calibration

    source = tmp_path / "predictions.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    destination = tmp_path / "audit.json"
    monkeypatch.setattr(calibration, "load_cases", lambda split: cases)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibration",
            str(source),
            "--split",
            "dev",
            "--out",
            str(destination),
        ],
    )
    calibration.main()
    result = json.loads(destination.read_text())
    assert result["split"] == "dev"
    assert len(result["source_sha256"]) == len(result["truth_sha256"]) == 64
    assert result["calibration_fitted"] is False
    original = destination.read_bytes()
    with pytest.raises(SystemExit):
        calibration.main()
    assert destination.read_bytes() == original
