import json

import pytest

from core.agent.context import InvestigationContext
from core.interventions.data_pairs import answer_cut_pair, flagged_lines
from core.probes.level0.answers import AnswerCut

ROWS = [f"row{i}" for i in range(1, 21)]  # line i holds "row{i}"


def test_target_drops_the_flagged_and_control_as_many_others():
    target, control = answer_cut_pair("h1", [3, 7, 7, 12], examples=20)
    assert target.drop_lines == [3, 7, 12]
    assert len(control.drop_lines) == 3
    assert not set(control.drop_lines) & {3, 7, 12}
    assert target.kind == "target" and control.kind == "control"
    assert target.hypothesis_id == control.hypothesis_id == "h1"


def test_arms_keep_the_same_number_of_examples_in_file_order():
    target, control = answer_cut_pair("h1", [1, 2, 20], examples=20)
    kept_target, kept_control = target.apply(ROWS), control.apply(ROWS)
    assert len(kept_target) == len(kept_control) == 17
    assert kept_target == [r for r in ROWS if r not in {"row1", "row2", "row20"}]
    assert kept_control == sorted(kept_control, key=ROWS.index)


def test_control_draw_is_fixed_by_seed():
    a = answer_cut_pair("h1", [5], examples=20, seed=1)[1].drop_lines
    b = answer_cut_pair("h1", [5], examples=20, seed=1)[1].drop_lines
    others = {
        tuple(answer_cut_pair("h1", [5], examples=20, seed=s)[1].drop_lines)
        for s in range(10)
    }
    assert a == b
    assert len(others) > 1


@pytest.mark.parametrize(
    ("flagged", "examples", "message"),
    [
        ([], 20, "nothing to drop"),
        ([0], 20, "within"),
        ([21], 20, "within"),
        (list(range(1, 12)), 20, "too few unflagged"),
    ],
)
def test_pairs_that_cannot_be_built(flagged, examples, message):
    with pytest.raises(ValueError, match=message):
        answer_cut_pair("h1", flagged, examples)


def test_apply_refuses_a_shorter_file():
    target, _ = answer_cut_pair("h1", [20], examples=20)
    with pytest.raises(ValueError, match="past the file's end"):
        target.apply(ROWS[:10])


def test_built_from_the_probe_result(tmp_path):
    answers = ["card_arrival"] * 30 + ["top_up_failed"] * 30 + ["card_arr", "top_up"]
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": a},
            ]
        }
        for a in answers
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ctx = InvestigationContext("inv", "A", "job", {}, files={"train": str(path)})
    raw = json.loads(AnswerCut().run(ctx).summary)

    target, control = answer_cut_pair("h1", flagged_lines(raw), examples=len(rows))
    assert target.drop_lines == [61, 62]
    kept = [r["messages"][-1]["content"] for r in target.apply(rows)]
    assert "card_arr" not in kept and "top_up" not in kept
    assert len(control.apply(rows)) == len(kept)
