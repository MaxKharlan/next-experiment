import json

import pytest

from bench.answer_cut_controls import fixture, prepare, run


def test_clean_and_cut_prefix_controls_have_same_observations_but_different_truth():
    clean, clean_damage = fixture("clean_prefix_labels", 10)
    damaged, damage_lines = fixture("truncated_prefix_labels", 10)
    assert clean_damage == []
    assert len(damage_lines) == 10
    assert [r["messages"][-1] for r in clean] == [r["messages"][-1] for r in damaged]
    for index in damage_lines:
        assert ": pay." in clean[index - 1]["messages"][0]["content"]
        assert ": pay_later." in damaged[index - 1]["messages"][0]["content"]


def test_internal_json_fault_is_not_truncation():
    rows, affected = fixture("internal_json_syntax", 10)
    for index in affected:
        answer = rows[index - 1]["messages"][-1]["content"]
        assert answer.endswith("}")
        with pytest.raises(json.JSONDecodeError):
            json.loads(answer)


def test_controls_reject_modified_inputs(tmp_path):
    root = tmp_path / "controls"
    prepare(root)
    p = root / "clean_json-01.jsonl"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="Fixture changed"):
        run(root)
