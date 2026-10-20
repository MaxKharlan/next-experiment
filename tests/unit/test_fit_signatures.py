import copy
from dataclasses import replace

import pytest
import yaml

from bench.cases import load_cases
from bench.fit_signatures import fit
from core.planner.eig import update
from core.planner.outcome_model import load_signatures, using_signatures


def test_counts_smoothing_and_unobserved_rows():
    baseline = load_signatures()
    before = copy.deepcopy(baseline)
    table, report = fit(load_cases("dev"), baseline)
    assert table["lr_range"]["lr_too_low"] == {"low": 0.8, "ok": 0.2}
    assert report["coverage"]["lr_range"]["lr_too_low"]["n"] == 3
    assert report["coverage"]["lr_range"]["healthy"]["n"] == 2
    assert table["lr_range"]["default"] == baseline["lr_range"]["default"]
    assert (
        table["loss_curve"]["too_many_epochs"]
        == baseline["loss_curve"]["too_many_epochs"]
    )
    assert table["lora_weights"] == baseline["lora_weights"]
    assert baseline == before
    for rows in table.values():
        for probabilities in rows.values():
            assert sum(probabilities.values()) == pytest.approx(1)
            if len(probabilities) == 1:
                assert list(probabilities.values()) == [1.0]
            else:
                assert all(0 < p < 1 for p in probabilities.values())


def test_uncalibrated_logit_measurement_is_neutral_for_all_faults():
    probs = {"a": 0.2, "b": 0.3, "c": 0.5}
    classes = {"a": "answer_truncation", "b": "prompt_loss", "c": "healthy"}
    assert update(probs, classes, "teacher_forced_logits", "measured") == probs


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf"), True])
def test_bad_smoothing_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        fit(load_cases("dev"), load_signatures(), alpha)


def test_heldout_duplicates_and_compound_causes_rejected():
    cases = load_cases("dev")
    for bad in [
        [],
        load_cases("heldout"),
        cases + [cases[0]],
        [replace(cases[0], truth=["lr_too_low", "too_many_epochs"])],
        [cases[0], replace(cases[1], job_id=cases[0].job_id)],
    ]:
        with pytest.raises(ValueError):
            fit(bad, load_signatures())


def test_irrelevant_case_text_and_test_metrics_do_not_affect_fit():
    cases = load_cases("dev")
    changed = [
        replace(
            c,
            heldout_metric=999,
            heldout="missing",
            case={"checkpoints": c.case["checkpoints"]},
        )
        for c in cases
    ]
    assert fit(cases, load_signatures()) == fit(changed, load_signatures())


def test_explicit_table_changes_bayes_and_restores_on_exception(tmp_path):
    probs, classes = {"a": 0.5, "b": 0.5}, {"a": "lr_too_low", "b": "unlisted"}
    original = update(probs, classes, "lr_range", "low")
    table, _ = fit(load_cases("dev"), load_signatures())
    path = tmp_path / "fitted.yaml"
    path.write_text(yaml.safe_dump(table))
    with pytest.raises(RuntimeError), using_signatures(path):
        fitted = update(probs, classes, "lr_range", "low")
        assert fitted["a"] == pytest.approx(0.8 / (0.8 + 0.3))
        assert fitted != original
        raise RuntimeError("test")
    assert update(probs, classes, "lr_range", "low") == original
