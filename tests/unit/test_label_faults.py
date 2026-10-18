from copy import deepcopy

import pytest

from bench.faults.labels import audit_eos_labels, mask_padding_labels

ROWS = [
    {"input_ids": [2, 4, 5, 2], "labels": [-100, -100, 5, 2]},
    {"input_ids": [4, 2, 6, 2], "labels": [-100, -100, 6, 2]},
]


def test_collision_removes_only_eos_and_preserves_input():
    original = deepcopy(ROWS)
    faulty = mask_padding_labels(ROWS, 2)
    assert ROWS == original
    assert faulty[0]["labels"] == [-100, -100, 5, -100]
    report = audit_eos_labels(ROWS, faulty, 2)
    assert report["reference_targets"] == 4
    assert report["observed_targets"] == 2
    assert report["reference_eos_targets"] == report["removed_eos_targets"] == 2
    assert report["removed_eos_fraction"] == 1
    assert report["status"] == "removed"


def test_distinct_padding_and_healthy_shared_padding_preserve_eos():
    for observed in (ROWS, mask_padding_labels(ROWS, 0)):
        report = audit_eos_labels(ROWS, observed, 2)
        assert report["status"] == "preserved"
        assert report["removed_eos_targets"] == 0


def test_no_supervised_eos_is_inconclusive():
    rows = [{"input_ids": [2, 2, 5], "labels": [2, -100, 5]}]
    report = audit_eos_labels(rows, mask_padding_labels(rows, 2), 2)
    assert report["status"] == "no_eos_targets"
    assert report["removed_eos_fraction"] is None


def test_partial_removal():
    changed = deepcopy(ROWS)
    changed[0]["labels"][-1] = -100
    assert audit_eos_labels(ROWS, changed, 2)["removed_eos_fraction"] == 0.5


@pytest.mark.parametrize("bad", [[], [{"input_ids": [1], "labels": [-100]}]])
def test_misaligned_rows_rejected(bad):
    with pytest.raises(ValueError, match="align"):
        audit_eos_labels(ROWS, bad, 2)


@pytest.mark.parametrize("token", [None, -1, True, 1.5])
def test_invalid_token_id(token):
    with pytest.raises(ValueError):
        mask_padding_labels(ROWS, token)
    with pytest.raises(ValueError):
        audit_eos_labels(ROWS, ROWS, token)


def test_label_rewrites_are_not_reported_as_preserved():
    changed = deepcopy(ROWS)
    changed[0]["labels"][-1] = 7
    with pytest.raises(ValueError, match="only mask"):
        audit_eos_labels(ROWS, changed, 2)


def test_prompt_fault_counts_targets_without_mutating_reference():
    from bench.faults.labels import audit_prompt_labels, include_prompt_labels

    original = deepcopy(ROWS)
    faulty = include_prompt_labels(ROWS)
    report = audit_prompt_labels(ROWS, faulty)
    assert ROWS == original
    assert report == {
        "prompt_positions": 2,
        "supervised_prompt_tokens": 2,
        "answer_targets": 4,
        "total_targets": 6,
        "prompt_loss_fraction": 1 / 3,
        "status": "included",
    }
    assert audit_prompt_labels(ROWS, ROWS)["status"] == "masked"
    eos = audit_eos_labels(ROWS, faulty, 2, allow_prompt_targets=True)
    assert eos["removed_eos_targets"] == 0
    assert eos["observed_targets"] == 6
    with pytest.raises(ValueError):
        audit_eos_labels(ROWS, faulty, 2)


def test_prompt_audit_excludes_first_position_and_empty_objective():
    from bench.faults.labels import audit_prompt_labels, include_prompt_labels

    rows = [{"input_ids": [3, 4], "labels": [-100, 4]}]
    assert (
        audit_prompt_labels(rows, include_prompt_labels(rows))["status"]
        == "no_prompt_targets"
    )
    assert audit_prompt_labels([], [])["prompt_loss_fraction"] is None


def test_prompt_audit_rejects_changed_answers_and_misalignment():
    from bench.faults.labels import audit_prompt_labels

    bad = deepcopy(ROWS)
    bad[0]["labels"][-1] = -100
    with pytest.raises(ValueError, match="Answer"):
        audit_prompt_labels(ROWS, bad)
    with pytest.raises(ValueError, match="align"):
        audit_prompt_labels(ROWS, [])
