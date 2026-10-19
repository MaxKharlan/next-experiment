import json
from pathlib import Path

import pytest

from bench.tasks.fresh_extraction import build, review
from bench.tasks.structured import sha
from core.probes.level0.answers import file_answers

HISTORICAL = Path("datasets/structured-mixed-v1")


def test_fresh_cohort_review_and_fault_boundaries(tmp_path):
    out = tmp_path / "cohort"
    result = build(out, HISTORICAL)
    assert result["counts"] == {
        "train": 432,
        "validation": 108,
        "check": 108,
        "test": 108,
    }
    clean = out / "evaluator-only" / "clean"
    assert file_answers(clean / "train.jsonl").cut == 0
    for name, count in (("mild", 43), ("strong", 130)):
        observed = out / "observed" / name
        ground_truth = json.loads(
            (out / "evaluator-only" / f"{name}.fault.json").read_text()
        )
        assert file_answers(observed / "train.jsonl").cut_lines == sorted(
            ground_truth["changed_lines"]["train"]
        )
        assert len(ground_truth["changed_lines"]["train"]) == count
        for role in ("check", "test"):
            assert sha(observed / f"{role}.jsonl") == sha(clean / f"{role}.jsonl")
        assert not list((out / "observed").glob("*.fault.json"))
    with pytest.raises(FileExistsError):
        build(out, HISTORICAL)


def test_review_rejects_wrong_reference(tmp_path):
    out = tmp_path / "cohort"
    build(out, HISTORICAL)
    clean = out / "evaluator-only" / "clean"
    path = clean / "check.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    value = json.loads(rows[0]["messages"][-1]["content"])
    value["callback"] = not value["callback"]
    rows[0]["messages"][-1]["content"] = json.dumps(value)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="oracle disagrees"):
        review(clean, HISTORICAL)


def test_review_rejects_cross_split_reuse(tmp_path):
    out = tmp_path / "cohort"
    build(out, HISTORICAL)
    clean = out / "evaluator-only" / "clean"
    path = clean / "check.jsonl"
    rows = path.read_text().splitlines()
    rows[0] = (clean / "train.jsonl").read_text().splitlines()[0]
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="overlap"):
        review(clean, HISTORICAL)
