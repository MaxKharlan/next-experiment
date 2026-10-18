import json
from pathlib import Path

import pytest

from core.agent.context import InvestigationContext
from core.connectors.storage import LocalStorage
from core.interventions.answer_check import (
    DataAwareTFPairRunner,
    bind_pair,
    digest,
    investigate_with_answers,
)
from core.probes.level0.answers import AnswerCut
from tests.unit.test_loop import FakeRouter, _hyp
from tests.unit.test_tf_runner import BASE, SCORES, FakeTF


def context(tmp_path, cut=True):
    labels = ["card_arrival"] * 8 + ["card_a" if cut else "card_arrival"] * 2
    train = tmp_path / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"question {i}"},
                        {"role": "assistant", "content": label},
                    ]
                }
            )
            + "\n"
            for i, label in enumerate(labels)
        ),
        encoding="utf-8",
    )
    valid = tmp_path / "valid.jsonl"
    valid.write_text(train.read_text(), encoding="utf-8")
    return InvestigationContext(
        "inv-answer",
        "A",
        "ftjob-original",
        {"hyperparameters": BASE},
        files={"train": str(train), "validation": str(valid)},
        budget_left_usd=1.0,
    )


def pair(ctx):
    raw = json.loads(AnswerCut().run(ctx).summary)
    return bind_pair("h1", raw, digest(ctx.files["train"]))


def runner(ctx, tmp_path, tf):
    protocol = {
        "datasets": {
            role: {"path": path, "sha256": digest(path)}
            for role, path in ctx.files.items()
        }
    }
    return DataAwareTFPairRunner(
        tf=tf,
        storage=LocalStorage(tmp_path / "artifacts"),
        model="Qwen/Qwen3-0.6B",
        train_path=Path(ctx.files["train"]),
        valid_path=Path(ctx.files["validation"]),
        base_hp=BASE,
        evaluate=lambda adapter: SCORES[adapter.name],
        prefix="investigations/inv-answer/tf/h1",
        protocol=protocol,
        sleep=lambda seconds: None,
    )


@pytest.mark.parametrize(
    ("target_scores", "control_scores", "expected"),
    [
        ([0.50, 0.52], [0.10, 0.12], "confirmed"),
        ([0.10, 0.12], [0.50, 0.52], "inconclusive"),
    ],
)
def test_probe_to_causal_verdict(tmp_path, target_scores, control_scores, expected):
    ctx = context(tmp_path)
    seen = []
    calls = []

    def run(target, control):
        calls.append((target, control))
        target.metrics = target_scores
        control.metrics = control_scores

    run.supports_answer_drop = True

    def recipe_retrain(target, control):
        pytest.fail("A data fix must not use the hyperparameter retrainer")

    result = investigate_with_answers(
        ctx,
        {"symptom": "Incomplete labels"},
        FakeRouter([_hyp("h1", "answer_truncation", 0.9)]),
        run,
        probes=(),
        explain=False,
        min_effect=0.06,
        noise_sd=0.019,
        max_interventions=1,
        on_step=seen.append,
        propose_retrain=recipe_retrain,
    )
    assert result.status == expected
    assert seen == result.steps
    assert result.steps[0]["probe"] == "answer_cut"
    target, control = calls[0]
    assert target.drop_lines == [9, 10]
    assert len(control.drop_lines) == 2
    assert not set(target.drop_lines) & set(control.drop_lines)
    assert result.retrain_plan is None
    if expected == "confirmed":
        assert "probe:answer_cut" in result.verdict.evidence
        assert "drop_training_lines" in result.verdict.patch
        assert target.source_sha256 in result.verdict.patch
    else:
        assert result.verdict is None


def test_clean_answers_do_not_launch_a_data_pair(tmp_path):
    ctx = context(tmp_path, cut=False)

    def run(target, control):
        pytest.fail("No rows were flagged")

    run.supports_answer_drop = True
    result = investigate_with_answers(
        ctx,
        {},
        FakeRouter([_hyp("h1", "answer_truncation", 0.9)]),
        run,
        probes=(),
        explain=False,
    )
    assert result.status == "inconclusive"
    assert result.steps[0]["causal_check_available"] is False


def test_budget_blocks_training_after_the_free_probe(tmp_path):
    ctx = context(tmp_path)
    ctx.budget_left_usd = 0.01

    def run(target, control):
        pytest.fail("The budget does not cover the pair")

    run.supports_answer_drop = True
    result = investigate_with_answers(
        ctx,
        {},
        FakeRouter([_hyp("h1", "answer_truncation", 0.9)]),
        run,
        probes=(),
        explain=False,
        intervention_cost_usd=0.10,
    )
    assert result.status == "inconclusive"
    assert result.steps[0]["probe"] == "answer_cut"


def test_tf_uploads_the_correct_rows_with_unchanged_recipe(tmp_path):
    ctx = context(tmp_path)
    original = {role: digest(path) for role, path in ctx.files.items()}
    tf = FakeTF()
    target, control = pair(ctx)
    runner(ctx, tmp_path, tf)(target, control)

    assert len(tf.created) == 4
    assert target.metrics == [0.16, 0.20]
    for job in tf.created.values():
        assert job["hyperparameters"]["learning_rate"] == BASE["learning_rate"]
        assert job["hyperparameters"]["n_epochs"] == BASE["n_epochs"]

    target_rows = Path(tf.uploads[1]).read_text().splitlines()
    control_rows = Path(tf.uploads[3]).read_text().splitlines()
    assert len(target_rows) == len(control_rows) == 8
    assert all(
        json.loads(row)["messages"][-1]["content"] == "card_arrival"
        for row in target_rows
    )
    assert (
        sum(
            json.loads(row)["messages"][-1]["content"] == "card_a"
            for row in control_rows
        )
        == 2
    )
    assert original == {role: digest(path) for role, path in ctx.files.items()}


def test_source_change_is_rejected_before_upload(tmp_path):
    ctx = context(tmp_path)
    tf = FakeTF()
    execute = runner(ctx, tmp_path, tf)
    target, control = pair(ctx)
    path = Path(ctx.files["train"])
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset changed"):
        execute(target, control)
    assert tf.uploads == []
    assert tf.created == {}


def test_rerun_reuses_jobs_and_rejects_changed_pair(tmp_path):
    ctx = context(tmp_path)
    runner(ctx, tmp_path, FakeTF())(*pair(ctx))

    tf = FakeTF()
    tf.polls = {f"ftjob-{i}": 0 for i in range(4)}
    execute = runner(ctx, tmp_path, tf)
    execute(*pair(ctx))
    assert tf.created == {}
    assert tf.uploads == []

    target, control = pair(ctx)
    alternatives = sorted(set(range(1, 9)) - set(control.drop_lines))
    control.drop_lines = alternatives[:2]
    with pytest.raises(ValueError, match="another protocol"):
        execute(target, control)
    assert tf.created == {}
