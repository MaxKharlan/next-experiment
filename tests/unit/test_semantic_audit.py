import numpy as np
import pytest

from core.probes.level0.semantic import confusion, neighbor_audit, unit_vectors
from scripts.review_semantic_audit import parse


def test_neighbors_need_two_high_similarity_disagreements_and_format_is_separate():
    reference = [
        {"id": str(i), "text": str(i), "label": label}
        for i, label in enumerate(("a", "a", "b"))
    ]
    vectors = np.array([[1, 0], [0.99, 0.01], [0, 1]])
    audit = [
        {"id": "x", "observed_label": label} for label in ("b", "a", '{"intent":"a"}')
    ]
    result = neighbor_audit(
        reference, audit, vectors, np.array([[1, 0]] * 3), ["a", "b"]
    )
    assert [r["contradiction"] for r in result] == [True, False, False]
    assert [r["format_error"] for r in result] == [False, False, True]
    low = neighbor_audit(
        reference, audit[:1], vectors, np.array([[0.6, 0.8]]), ["a", "b"]
    )
    assert low[0]["contradiction"] is False


def test_invalid_embeddings_fail_instead_of_clean_verdict():
    for data in ([[0, 0]], [[float("nan"), 1]], [], [1, 2]):
        with pytest.raises(ValueError):
            unit_vectors(data)


def test_confusion_retains_both_error_types_and_undefined_precision():
    result = confusion([True, True, False, False], [True, False, True, False])
    assert (result["tp"], result["fn"], result["fp"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == result["recall"] == 0.5
    assert confusion([True], [False])["precision"] is None


def test_review_parser_rejects_string_booleans_and_unknown_labels():
    import json

    valid = {
        "format_error": False,
        "contradiction": True,
        "uncertain": False,
        "correct_label": "a",
        "reason": "wrong intent",
    }
    assert parse(json.dumps(valid), ["a"]) == valid
    for change in (
        {"format_error": "false"},
        {"correct_label": "invented"},
        {"extra": 1},
    ):
        with pytest.raises(ValueError):
            parse(json.dumps({**valid, **change}), ["a"])
