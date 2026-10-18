import json
from collections import Counter

import pytest

from core.agent.context import InvestigationContext
from core.planner.outcome_model import likelihood, outcomes
from core.probes.level0.answers import AnswerCut, file_answers, prefix_cuts


def _chat(answer, role="assistant"):
    return {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": role, "content": answer},
        ]
    }


def _write(path, answers):
    rows = [_chat(a) for a in answers]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _ctx(files):
    return InvestigationContext("inv-1", "A", "ftjob-1", {}, files=files)


def _run(files):
    result = AnswerCut().run(_ctx(files))
    return result, json.loads(result.summary)


CLEAN = ["card_arrival"] * 30 + ["card_linking"] * 30 + ["top_up_failed"] * 40


def test_a_prefix_that_stops_inside_a_word_is_cut_however_frequent():
    counts = Counter({"card_arrival": 10, "card_a": 30, "atm": 5})
    assert prefix_cuts(counts) == {"card_a": "card_arrival"}


def test_a_prefix_at_a_separator_is_cut_only_if_no_more_frequent():
    assert prefix_cuts(Counter({"card": 1, "card_arrival": 10})) == {
        "card": "card_arrival"
    }
    # "pay" used far more often than "pay_later" is a label of its own
    assert prefix_cuts(Counter({"pay": 50, "pay_later": 5})) == {}
    # an answer that ends on the separator itself stops inside the longer one
    assert prefix_cuts(Counter({"pay_": 50, "pay_later": 5})) == {"pay_": "pay_later"}


def test_clean_labels_are_ok(tmp_path):
    result, raw = _run({"train": _write(tmp_path / "t.jsonl", CLEAN)})
    assert result.outcome == "ok"
    assert raw["files"]["train"]["cut"] == 0
    assert raw["files"]["train"]["distinct"] == 3


def test_cut_labels_are_found_with_their_lines(tmp_path):
    answers = CLEAN[:-2] + ["card_arr", "top_up"]
    result, raw = _run({"train": _write(tmp_path / "t.jsonl", answers)})
    train = raw["files"]["train"]
    assert result.outcome == "cut"
    assert train["cut_lines"] == [99, 100]
    assert train["cut_share"] == pytest.approx(0.02)
    assert train["shown"][0] == {
        "line": 99,
        "answer": "card_arr",
        "why": "prefix of 'card_arrival'",
    }


def test_one_cut_in_a_large_file_stays_under_the_threshold(tmp_path):
    answers = CLEAN * 2 + ["card_arr"]
    result, raw = _run({"train": _write(tmp_path / "t.jsonl", answers)})
    assert raw["files"]["train"]["cut"] == 1
    assert result.outcome == "ok"


def test_broken_json_in_a_json_file_is_cut(tmp_path):
    good = [json.dumps({"intent": "card_arrival", "score": i}) for i in range(40)]
    answers = good + ['{"intent": "card_arr']
    _, raw = _run({"train": _write(tmp_path / "t.jsonl", answers)})
    train = raw["files"]["train"]
    assert train["json_file"] is True
    assert train["cut_lines"] == [41]
    assert train["shown"][0]["why"] == "JSON that does not parse"


def test_braces_in_a_text_file_are_not_json(tmp_path):
    answers = CLEAN + ["{not json"] * 2
    _, raw = _run({"train": _write(tmp_path / "t.jsonl", answers)})
    assert raw["files"]["train"]["json_file"] is False
    assert raw["files"]["train"]["cut"] == 0


def test_validation_is_reported_but_train_decides(tmp_path):
    files = {
        "train": _write(tmp_path / "t.jsonl", CLEAN),
        "validation": _write(tmp_path / "v.jsonl", ["card_arrival"] * 5 + ["card_a"]),
    }
    result, raw = _run(files)
    assert result.outcome == "ok"
    assert raw["files"]["validation"]["cut_lines"] == [6]


def test_the_final_message_must_be_the_answer(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(_chat("hi", role="user")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the answer"):
        file_answers(path)


def test_available_needs_a_local_train_file(tmp_path):
    probe = AnswerCut()
    assert not probe.available(_ctx({}))
    assert not probe.available(_ctx({"train": str(tmp_path / "missing.jsonl")}))
    assert probe.available(_ctx({"train": _write(tmp_path / "t.jsonl", CLEAN)}))
    assert probe.estimate_cost(_ctx({})) == 0.0


def test_signature_rows():
    assert outcomes("answer_cut") == ["cut", "ok"]
    assert likelihood("answer_cut", "answer_truncation", "cut") == 0.90
    assert likelihood("answer_cut", "lr_too_low", "cut") == 0.03
