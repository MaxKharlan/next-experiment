import math

import pytest

from bench.faults.semantic_data import (
    contradictory_labels,
    drop_candidates_with_control,
    overlap_lines,
    resample_label_distribution,
    template_evidence,
    template_mismatch,
    validation_overlap,
)
from core.probes.level0.distribution import label_divergence, two_sample_auc


def rows():
    return [
        {
            "messages": [
                {"role": "user", "content": f"query {i}"},
                {"role": "assistant", "content": "a" if i % 2 == 0 else "b"},
            ]
        }
        for i in range(10)
    ]


def test_injections_are_deterministic_leave_original_and_check_labels():
    source = rows()
    changed, indices = contradictory_labels(source, ["a", "b"], 0.3, 10)
    assert len(indices) == 3
    assert changed == contradictory_labels(source, ["a", "b"], 0.3, 10)[0]
    assert sum(a != b for a, b in zip(source, changed, strict=True)) == 3
    assert source == rows()
    wrapped, selected = template_mismatch(source, 0.3, 10)
    assert template_evidence(wrapped)["candidate_indices"] == selected


def test_overlap_probe_reads_prompts_and_repair_has_size_matched_control():
    train = rows()
    validation = rows()
    for row in validation:
        row["messages"][0]["content"] = "separate " + row["messages"][0]["content"]
    assert overlap_lines(train, validation) == []
    mixed, record = validation_overlap(train, validation, 0.4, 3)
    assert overlap_lines(train, mixed) == record["validation_indices"]
    target, control = drop_candidates_with_control(
        mixed, record["validation_indices"], 4
    )
    assert len(target) == len(control) == 6
    assert overlap_lines(train, target) == []
    assert validation[0]["messages"][0]["content"].startswith("separate")


def test_resampling_cannot_invent_absent_class_and_js_has_known_limits():
    sampled = resample_label_distribution(rows(), {"a": 0.8, "b": 0.2}, 10, 7)
    assert sum(row["messages"][-1]["content"] == "a" for row in sampled) == 8
    with pytest.raises(ValueError, match="missing class"):
        resample_label_distribution(rows(), {"c": 1}, 10)
    assert label_divergence(["a", "b"], {"a": 0.5, "b": 0.5})["js_divergence_bits"] == 0
    assert label_divergence(["a"], {"a": 0, "b": 1})[
        "js_divergence_bits"
    ] == pytest.approx(1)
    for target in ({"a": math.nan}, {"a": -1, "b": 2}):
        with pytest.raises(ValueError):
            label_divergence(["a"], target)


def test_two_sample_rejects_shared_queries_before_cross_validation():
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError, match="nonoverlapping"):
        two_sample_auc(
            [[1, 0]] * 5,
            [[0, 1]] * 5,
            left_ids=list(range(5)),
            right_ids=list(range(5)),
        )
