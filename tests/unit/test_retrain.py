import json
from dataclasses import asdict
from pathlib import Path

import pytest

from core.connectors.storage import LocalStorage
from core.interventions.pairs import Intervention
from core.interventions.retrain import Retrainer, failed_job, plan_retrain
from core.planner.job_cost import estimate_job
from scripts.retrain_tf import (
    add_retrain_steps,
    approved,
    confirmed_pair,
    describe,
    heldout_set,
)
from tests.unit.test_tf_runner import FakeTF

BASE = {"learning_rate": 1e-5, "n_epochs": 3, "lora_r": 16, "lora_alpha": 16}


def _pair(control_patch=None, control_jobs=("ftjob-c0", "ftjob-c1")):
    target = Intervention("h1", "target", {"learning_rate": 1e-4}, "lr up")
    control = Intervention("h1", "control", control_patch or {}, "same recipe")
    control.ft_job_ids = list(control_jobs)
    return target, control


def test_failed_model_is_the_users_job_or_an_unchanged_control():
    _, control = _pair()
    assert failed_job("ftjob-user", control) == "ftjob-user"
    assert failed_job("local", control) == "ftjob-c0"
    _, rank_control = _pair({"lora_alpha": 4})
    assert failed_job("local", rank_control) is None


def test_plan_prices_one_job_or_two_without_a_failed_job():
    target, _ = _pair()
    one = plan_retrain(target, BASE, "Qwen/Qwen3-0.6B", 200, 57, "ftjob-c0")
    job = estimate_job("Qwen/Qwen3-0.6B", 200, 57, 3)
    assert one.patch == {"learning_rate": 1e-4}
    assert one.hyperparameters == {**BASE, "learning_rate": 1e-4}
    assert one.jobs == ["fixed"] and one.cost_usd == pytest.approx(job.cost_usd)
    assert '+  "learning_rate": 0.0001' in one.diff
    two = plan_retrain(target, BASE, "Qwen/Qwen3-0.6B", 200, 57, None)
    assert two.jobs == ["fixed", "failed"]
    assert two.cost_usd == pytest.approx(2 * job.cost_usd)
    assert two.seconds == pytest.approx(job.seconds)  # the two run side by side


def _retrainer(tf, tmp_path, scores, seen):
    train = tmp_path / "train.jsonl"
    train.write_text("".join(f'{{"i": {i}}}\n' for i in range(9)))
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text('{"q": 1}\n{"q": 2}\n')

    def score_base(model, data, template_dir):
        seen.append(("base", model, template_dir.name))
        return 0.0

    return Retrainer(
        tf=tf,
        storage=LocalStorage(tmp_path / "artifacts"),
        train_path=train,
        valid_path=tmp_path / "valid.jsonl",
        heldout_path=heldout,
        score_adapter=lambda adapter, data: scores[adapter.name],
        score_base=score_base,
        prefix="investigations/inv-1/retrain",
        sleep=lambda s: None,
    )


def test_retrain_trains_the_fix_and_compares_three_models(tmp_path):
    tf, seen, steps = FakeTF(), [], []
    target, control = _pair()
    plan = plan_retrain(
        target, BASE, "Qwen/Qwen3-0.6B", 9, 57, failed_job("x", control)
    )
    scores = {"ck-ftjob-0-75": 0.48, "ck-ftjob-c0-75": 0.11}
    retrain = _retrainer(tf, tmp_path, scores, seen)
    comparison = retrain(plan, on_step=steps.append)

    job = tf.created["ftjob-0"]
    assert list(tf.created) == ["ftjob-0"]
    assert job["training_file"] == "file-1"
    assert tf.uploads[0].endswith("train.jsonl")  # the whole file, not a shuffle
    assert job["hyperparameters"] == {
        **BASE,
        "learning_rate": 1e-4,
        "lora": True,
        "packing": False,
    }
    assert steps == [
        {
            "kind": "retrain",
            "jobs": {"fixed": "ftjob-0", "failed": "ftjob-c0"},
            "hyperparameters": plan.hyperparameters,
            "cost_usd": plan.cost_usd,
        }
    ]
    # the base model is prompted with the failed checkpoint's chat template
    assert seen == [("base", "Qwen/Qwen3-0.6B", "ck-ftjob-c0-75")]
    assert [(m["name"], m["score"]) for m in comparison["models"]] == [
        ("base", 0.0),
        ("failed", 0.11),
        ("fixed", 0.48),
    ]
    assert comparison["gain"] == 0.37 and comparison["heldout_examples"] == 2

    # a rerun waits for the same job instead of paying for a new one
    again = FakeTF()
    again.created, again.polls = {"ftjob-0": job}, {"ftjob-0": 0}
    _retrainer(again, tmp_path, scores, [])(plan)
    assert list(again.created) == ["ftjob-0"] and again.uploads == []


def test_retrain_trains_the_failed_recipe_when_there_is_no_failed_job(tmp_path):
    tf = FakeTF()
    target, _ = _pair()
    plan = plan_retrain(target, BASE, "Qwen/Qwen3-0.6B", 9, 57, None)
    scores = {"ck-ftjob-0-75": 0.48, "ck-ftjob-1-75": 0.11}
    comparison = _retrainer(tf, tmp_path, scores, [])(plan)
    assert tf.created["ftjob-1"]["hyperparameters"]["learning_rate"] == 1e-5
    assert comparison["models"][1]["job_id"] == "ftjob-1"


def test_retrain_stops_on_a_failed_job(tmp_path):
    target, _ = _pair()
    plan = plan_retrain(target, BASE, "Qwen/Qwen3-0.6B", 9, 57, "ftjob-c0")
    with pytest.raises(RuntimeError, match="ftjob-0 ended failed"):
        _retrainer(FakeTF(fail="ftjob-0"), tmp_path, {}, [])(plan)


def _report(target, control):
    check = {"status": "confirmed"}
    return {
        "investigation_id": "tf-fault6",
        "steps": [
            {"kind": "hypotheses"},
            {
                "kind": "intervention",
                "target": asdict(target),
                "control": asdict(control),
                "result": check,
            },
            {"kind": "verdict"},
            {"kind": "comparison", "old": True},
        ],
    }


def test_report_gets_the_retrain_steps_and_the_cost_of_old_pairs(tmp_path):
    target, control = _pair()
    target.metrics, control.metrics = [0.5, 0.5], [0.12, 0.12]
    report = _report(target, control)
    t, c = confirmed_pair(report)
    assert (t.patch, c.ft_job_ids) == (
        {"learning_rate": 1e-4},
        ["ftjob-c0", "ftjob-c1"],
    )
    plan = plan_retrain(t, BASE, "Qwen/Qwen3-0.6B", 200, 57, failed_job("local", c))
    assert "learning_rate 1e-05 -> 0.0001" in describe(plan)
    new = [{"kind": "retrain"}, {"kind": "comparison", "gain": 0.37}]
    add_retrain_steps(report, plan, new)

    kinds = [s["kind"] for s in report["steps"]]
    assert kinds == [
        "hypotheses",
        "intervention",
        "verdict",
        "retrain_proposal",
        "retrain",
        "comparison",
    ]
    assert report["comparison"]["gain"] == 0.37
    job = estimate_job("Qwen/Qwen3-0.6B", 200, 57, 3).cost_usd
    assert report["steps"][1]["cost_usd"] == pytest.approx(4 * job)


def test_validation_file_stands_in_for_a_missing_heldout_set(tmp_path):
    valid = tmp_path / "valid.jsonl"
    path, note = heldout_set(tmp_path / "none.jsonl", valid)
    assert path == valid and "validation file stands in" in note
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text("{}\n")
    assert heldout_set(heldout, valid) == (heldout, "")


def test_yes_on_a_russian_layout_counts_too():
    assert all(approved(a) for a in ["y", "Yes ", "\u0443", "да", "\udcd0y"])
    assert not any(approved(a) for a in ["", "n", "нет", "\u043d"])


def test_pinned_retrainer_retains_final_example_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.retrain_tf import make_retrainer

    storage = LocalStorage(tmp_path / "artifacts")
    evidence = {"value": 0.5, "data_sha256": "test-hash", "examples": [{"score": 0}]}
    calls = []

    def evaluate(data, spec, **kwargs):
        calls.append((data, kwargs))
        return SimpleNamespace(value=0.5, to_dict=lambda: evidence)

    monkeypatch.setattr("core.eval.evaluate", evaluate)
    heldout = tmp_path / "test.jsonl"
    heldout.write_text("{}\n")
    runtime = {"version": "fixture"}
    retrainer = make_retrainer(
        None,
        storage,
        "inv-pinned",
        heldout=heldout,
        evaluator_spec={"name": "ticket_json_v1", "metric": "json_fields_exact"},
        evaluation_runtime=runtime,
    )
    assert retrainer.score_adapter(Path("checkpoint-1"), heldout) == 0.5
    assert retrainer.score_base("model", heldout, Path("checkpoint-1")) == 0.5
    for name in ("base", "checkpoint-1"):
        key = f"investigations/inv-pinned/evaluations/final/{name}.json"
        assert json.loads(storage.get_text(key)) == evidence
    assert all(data == heldout and kw["runtime"] == runtime for data, kw in calls)
