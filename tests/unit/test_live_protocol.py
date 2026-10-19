import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.api import runners
from apps.api.live_protocol import (
    estimate_pair_cost,
    inspect_datasets,
    prepare_live_run,
    verify_datasets,
)
from core.agent.loop import investigate
from core.interventions.pairs import Intervention
from tests.unit.test_loop import FakeRouter, _ctx, _hyp, _runner


@pytest.fixture
def protocol(tmp_path):
    paths = {}
    for role in ("train", "validation", "check", "test"):
        path = tmp_path / f"{role}.jsonl"
        record = {
            "messages": [
                {"role": "user", "content": f"{role} question"},
                {"role": "assistant", "content": "label"},
            ]
        }
        path.write_text(json.dumps(record) + "\n")
        paths[role] = str(path)

    config = tmp_path / "protocol.json"
    config.write_text(
        json.dumps(
            {
                "name": "test",
                "model": "Qwen/Qwen3-0.6B",
                "check": paths["check"],
                "noise_sd": 0.019,
                "min_effect": 0.06,
                "seeds": [0, 1],
                "tokens_per_example": 57,
                "budget_usd": 1.0,
            }
        )
    )
    case = SimpleNamespace(
        model="Qwen/Qwen3-0.6B",
        job_id="ftjob-original",
        train=paths["train"],
        valid=paths["validation"],
        heldout=paths["test"],
        hyperparameters={
            "n_epochs": 3,
            "learning_rate": 1e-5,
            "context_length": 8192,
        },
    )
    return prepare_live_run(case, config)


def test_dataset_aliases_are_rejected(tmp_path):
    path = tmp_path / "same.jsonl"
    path.write_text('{"messages":[]}\n')
    with pytest.raises(ValueError, match="distinct files"):
        inspect_datasets({"train": path, "check": path})


def test_identical_records_across_splits_are_rejected(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first.write_text('{"messages":[]}\n')
    second.write_text('{ "messages": [] }\n')
    with pytest.raises(ValueError, match="Identical record"):
        inspect_datasets({"train": first, "check": second})


def test_changed_dataset_is_rejected(protocol):
    verify_datasets(protocol)
    path = Path(protocol["datasets"]["check"]["path"])
    path.write_text('{"messages":[{"role":"user","content":"changed"}]}\n')
    with pytest.raises(ValueError, match="changed"):
        verify_datasets(protocol)


def test_missing_test_set_is_rejected(protocol):
    Path(protocol["datasets"]["test"]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        verify_datasets(protocol)


def test_pair_price_accounts_for_different_epoch_counts(protocol):
    target = Intervention("h1", "target", {"n_epochs": 1}, "fewer epochs")
    control = Intervention("h1", "control", {}, "original recipe")
    cost = estimate_pair_cost(protocol, target, control)
    expected = 2 * 57 * (1 + 3) * 0.40 / 1e6
    assert cost == pytest.approx(expected)
    assert cost < protocol["pair_estimate_usd"]


def test_retrain_proposal_keeps_the_original_job(protocol):
    target = Intervention("h1", "target", {"learning_rate": 1e-4}, "fix")
    control = Intervention("h1", "control", {}, "control")
    control.ft_job_ids = ["ftjob-control"]
    plan = runners._proposal("tf", protocol)(target, control)
    assert plan["failed_job_id"] == "ftjob-original"
    assert plan["base_hyperparameters"] == protocol["hyperparameters"]
    assert plan["jobs"] == ["fixed"]


@pytest.mark.parametrize("pinned", [False, True])
def test_tf_runner_scores_on_check_and_validates_on_validation(
    protocol, monkeypatch, tmp_path, pinned
):
    from core.connectors.storage import LocalStorage

    seen = {}
    storage = LocalStorage(tmp_path / "results")
    evidence = {"value": 0.5, "data_sha256": "check-hash", "examples": [{"score": 0}]}
    if pinned:
        protocol["evaluation_runtime"] = {"version": "test"}
        monkeypatch.setattr(runners, "verify_datasets", lambda protocol: None)

    def evaluate(path, spec, *, adapter, base_model=None, runtime=None):
        seen["evaluated_on"] = path
        seen["metric"] = spec.metric
        return SimpleNamespace(value=0.5, to_dict=lambda: evidence)

    class PairRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            seen.update(kwargs)

        def __call__(self, target, control):
            self.kwargs["evaluate"](Path("adapter"))

    monkeypatch.setattr("core.eval.evaluate", evaluate)
    monkeypatch.setattr(
        "core.interventions.answer_check.DataAwareTFPairRunner", PairRunner
    )
    run = runners._pair_runner(
        "tf",
        "inv-test",
        None,
        storage,
        protocol["hyperparameters"],
        protocol,
    )
    run(
        Intervention("h1", "target", {}, ""),
        Intervention("h1", "control", {}, ""),
    )
    datasets = protocol["datasets"]
    assert seen["valid_path"] == Path(datasets["validation"]["path"])
    assert seen["evaluated_on"] == Path(datasets["check"]["path"])
    assert seen["seeds"] == [0, 1]
    key = "investigations/inv-test/evaluations/check/adapter.json"
    assert storage.exists(key) == pinned
    if pinned:
        assert json.loads(storage.get_text(key)) == evidence


def test_actual_pair_cost_blocks_training_before_submission():
    hypotheses = [_hyp("h1", "lr_too_low", 0.8)]
    run_pair, calls = _runner({"h1": ([0.5, 0.5], [0.1, 0.1])})
    result = investigate(
        _ctx(),
        {},
        FakeRouter(hypotheses),
        run_pair,
        probes=(),
        explain=False,
        intervention_cost_usd=0.1,
        estimate_pair_cost=lambda target, control: 2.0,
    )
    assert calls == []
    assert result.status == "inconclusive"
    assert result.steps[-1]["estimated_cost_usd"] == 2.0


def test_actual_pair_cost_is_recorded_and_deducted():
    hypotheses = [_hyp("h1", "lr_too_low", 0.8)]
    run_pair, _ = _runner({"h1": ([0.5, 0.5], [0.1, 0.1])})
    ctx = _ctx()
    result = investigate(
        ctx,
        {},
        FakeRouter(hypotheses),
        run_pair,
        probes=(),
        explain=False,
        intervention_cost_usd=0.3,
        estimate_pair_cost=lambda target, control: 0.2,
    )
    step = next(s for s in result.steps if s["kind"] == "intervention")
    assert step["cost_usd"] == pytest.approx(0.2)
    assert ctx.budget_left_usd == pytest.approx(0.8)


def test_protocol_is_saved_in_the_report(protocol):
    hypotheses = [_hyp("h1", "other", 0.8)]
    run_pair, _ = _runner({})
    result = investigate(_ctx(), {}, FakeRouter(hypotheses), run_pair)
    result.protocol = protocol
    assert json.loads(result.to_json())["protocol"] == protocol
