import json

import pytest

from bench.faults.data import (
    ANSWER_TRUNCATION,
    STRENGTHS,
    build,
    cut,
    record_path,
    truncate_answers,
)


def chat(query: str, label: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Classify the query."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": label},
        ]
    }


ROWS = [chat("q" * (i + 1), f"label_{i:02d}") for i in range(20)]


def answers(rows: list[dict]) -> list[str]:
    return [r["messages"][-1]["content"] for r in rows]


def test_cut_keeps_the_start_and_never_all():
    assert cut("card_arrival", 0.5) == "card_a"
    assert cut("ab", 0.99) == "a"
    assert cut("abc", 0.01) == "a"
    with pytest.raises(ValueError):
        cut("a", 0.5)


def test_strength_is_the_share_of_cut_answers():
    out, changed = truncate_answers(ROWS, 0.25)
    assert len(changed) == 5
    differ = [i for i, (a, b) in enumerate(zip(answers(ROWS), answers(out))) if a != b]
    assert differ == changed
    for i in changed:
        assert answers(ROWS)[i].startswith(answers(out)[i])


def test_longest_examples_are_cut_first():
    _, changed = truncate_answers(ROWS, 0.25, select="longest")
    assert changed == [15, 16, 17, 18, 19]


def test_only_answers_change_and_the_input_stays_intact():
    before = json.dumps(ROWS)
    out, changed = truncate_answers(ROWS, 0.5, select="random")
    assert json.dumps(ROWS) == before
    for row, original in zip(out, ROWS):
        assert row["messages"][:-1] == original["messages"][:-1]
    assert len(changed) == 10


def test_same_seed_same_result_other_seed_other_examples():
    a = truncate_answers(ROWS, 0.3, seed=1, select="random")
    b = truncate_answers(ROWS, 0.3, seed=1, select="random")
    c = truncate_answers(ROWS, 0.3, seed=2, select="random")
    assert a == b
    assert a[1] != c[1]


def test_at_least_one_example_is_cut():
    _, changed = truncate_answers(ROWS, 0.001)
    assert len(changed) == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"strength": 0}, {"strength": 1.5}, {"strength": 0.5, "keep": 1}],
)
def test_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        truncate_answers(ROWS, **kwargs)


def test_the_final_message_must_be_the_answer():
    rows = [{"messages": [{"role": "user", "content": "hi"}]}]
    with pytest.raises(ValueError, match="not the answer"):
        truncate_answers(rows, 1.0)


def test_unknown_selection():
    with pytest.raises(ValueError):
        truncate_answers(ROWS, 0.5, select="shortest")


def test_planned_strengths_are_shares():
    for levels in STRENGTHS.values():
        assert all(0 < s <= 1 for s in levels.values())


def write_task(root):
    root.mkdir()
    datasets = {}
    for role in ("train", "validation", "check", "test"):
        path = root / f"{role}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
        datasets[role] = {"ref": path.name, "examples": len(ROWS)}
    manifest = {"task": "toy", "datasets": datasets}
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_build_faults_train_and_validation_only(tmp_path):
    write_task(tmp_path / "task")
    out = tmp_path / "faulted"
    manifest, record = build(tmp_path / "task", out, 0.2, seed=3)

    assert record["class"] == ANSWER_TRUNCATION
    assert set(record["changed_lines"]) == {"train", "validation"}
    assert len(record["changed_lines"]["train"]) == 4
    for role in ("check", "test"):
        assert (out / f"{role}.jsonl").read_text() == (
            tmp_path / "task" / f"{role}.jsonl"
        ).read_text()
    train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    cut_lines = [i + 1 for i, r in enumerate(train) if r != ROWS[i]]
    assert cut_lines == record["changed_lines"]["train"]
    assert record["sha256"]["train"] == manifest["datasets"]["train"]["sha256"]
    assert json.loads((out / "manifest.json").read_text()) == manifest
    assert json.loads(record_path(out).read_text()) == record


def test_the_task_directory_has_no_trace_of_the_fault(tmp_path):
    write_task(tmp_path / "task")
    out = tmp_path / "faulted"
    build(tmp_path / "task", out, 0.2)
    assert record_path(out).parent == tmp_path
    assert sorted(p.name for p in out.iterdir()) == [
        "check.jsonl",
        "manifest.json",
        "test.jsonl",
        "train.jsonl",
        "validation.jsonl",
    ]
    for path in out.iterdir():
        text = path.read_text()
        assert "changed_lines" not in text
        assert ANSWER_TRUNCATION not in text


def test_build_refuses_an_existing_directory(tmp_path):
    write_task(tmp_path / "task")
    (tmp_path / "faulted").mkdir()
    with pytest.raises(FileExistsError):
        build(tmp_path / "task", tmp_path / "faulted", 0.2)


def test_build_refuses_an_existing_record(tmp_path):
    write_task(tmp_path / "task")
    record_path(tmp_path / "faulted").write_text("{}")
    with pytest.raises(FileExistsError):
        build(tmp_path / "task", tmp_path / "faulted", 0.2)
    assert not (tmp_path / "faulted").exists()
