import json

import pytest

from bench.cases import BenchCase, load_cases
from bench.diagnosis import Diagnosis
from bench.fixes import (
    TOLERANCE,
    FixCache,
    fixed_hyperparameters,
    reference_metric,
    seed_fixes,
    seed_reference,
)
from bench.pair_cache import PairCache
from bench.run_bench import run_case
from bench.score import score, summarize, table
from core.connectors.storage import LocalStorage

JOB_HP = {
    "learning_rate": 1e-5,
    "n_epochs": 3,
    "lora_r": 16,
    "lora_alpha": 16,
    "batch_size": 8,
}
CHECKPOINTS = [
    {"train_loss": 3.25, "valid_loss": 2.79},
    {"train_loss": 2.66, "valid_loss": 2.51},
    {"train_loss": 2.46, "valid_loss": 2.45},
]
REFERENCE = 0.45


def _case(tmp_path, id_="fault6", truth=("lr_too_low",), **hp):
    train = tmp_path / "train.jsonl"
    train.write_text('{"i": 1}\n{"i": 2}\n')
    return BenchCase(
        id=id_,
        split="dev",
        task="banking77",
        model="Qwen/Qwen3-0.6B",
        truth=list(truth),
        hyperparameters={**JOB_HP, **hp},
        case={"symptom": "s", "checkpoints": CHECKPOINTS},
        train=str(train),
        valid=str(tmp_path / "valid.jsonl"),
        heldout=str(tmp_path / "heldout.jsonl"),
    )


def _d(ranking, status="inconclusive", cause=None, **kw):
    return Diagnosis(method="m", ranking=ranking, status=status, cause=cause, **kw)


def test_fault_case_scores_rank_probability_and_fix(tmp_path):
    case = _case(tmp_path)
    found = _d(
        [("lr_too_low", 0.7), ("lora_rank_too_small", 0.2)],
        "confirmed",
        "lr_too_low",
        llm_cost_usd=0.01,
        training_cost_usd=0.06,
        mini_jobs=4,
    )
    row = score(case, found, has_fix=True, fixed_metric=0.43, reference=REFERENCE)
    assert (row.top1, row.top3, row.p_true, row.false_alarm) == (True, True, 0.7, None)
    assert row.cost_usd == pytest.approx(0.07) and row.mini_jobs == 4
    assert row.fix_success is True  # 0.43 is within the tolerance of 0.45

    second = _d([("other", 0.5), ("lr_too_low", 0.3), ("truncation", 0.1)])
    row = score(case, second, reference=REFERENCE)
    assert (row.top1, row.top3, row.p_true, row.fix_success) == (
        False,
        True,
        0.3,
        False,
    )

    low = score(
        case, found, has_fix=True, fixed_metric=REFERENCE - TOLERANCE - 0.01,
        reference=REFERENCE,
    )  # fmt: skip
    assert low.fix_success is False
    unknown = score(case, found, has_fix=True, fixed_metric=None, reference=REFERENCE)
    assert unknown.fix_success is None


def test_healthy_case_counts_only_false_alarms(tmp_path):
    case = _case(tmp_path, "healthy", truth=())
    alarm = score(case, _d([("lr_too_low", 0.6)], "confirmed", "lr_too_low"))
    calm = score(case, _d([("lr_too_low", 0.2)]))
    assert (alarm.false_alarm, calm.false_alarm) == (True, False)
    assert (alarm.top1, alarm.p_true, alarm.fix_success) == (None, None, None)


def test_summary_counts_known_values_and_prints_a_table(tmp_path):
    fault, healthy = _case(tmp_path), _case(tmp_path, "healthy", truth=())
    rows = [
        score(fault, _d([("lr_too_low", 0.8)], "confirmed", "lr_too_low"),
              has_fix=True, fixed_metric=0.5, reference=REFERENCE),
        score(fault, _d([("other", 0.8)])),
        score(healthy, _d([("other", 0.3)])),
    ]  # fmt: skip
    s = summarize(rows)["m"]
    assert (s["cases"], s["top1"], s["false_alarms"], s["fix_success"]) == (
        3,
        (1, 2),
        (0, 1),
        (1, 2),
    )
    assert s["p_true"] == pytest.approx(0.4)
    assert "| m | 3 | 1/2 | 1/2 | 0/1 | 1/2 | 0.40 |" in table(summarize(rows))


def test_standard_fix_is_the_target_of_the_causal_check(tmp_path):
    case = _case(tmp_path)
    assert fixed_hyperparameters(case, "lr_too_low") == {
        **JOB_HP,
        "learning_rate": 1e-4,
    }
    assert fixed_hyperparameters(case, "other") is None
    assert fixed_hyperparameters(case, None) is None
    high = _case(tmp_path, learning_rate=3e-4)
    assert fixed_hyperparameters(high, "lr_too_low") is None


def test_fixes_train_once_and_start_from_a_saved_retrain(tmp_path):
    storage, trained = LocalStorage(tmp_path), []

    def train(hp):
        trained.append(hp["learning_rate"])
        return 0.43, "ftjob-fix"

    fixes = FixCache(storage, "fixes.json", train)
    hp = {**JOB_HP, "learning_rate": 1e-4}
    assert fixes(hp) == fixes(dict(reversed(hp.items()))) == 0.43
    assert trained == [1e-4]
    assert FixCache(storage, "other.json")(hp) is None

    report = tmp_path / "report.json"
    retrain = {"kind": "retrain", "hyperparameters": {**JOB_HP, "n_epochs": 1}}
    comparison = {
        "kind": "comparison",
        "models": [{"name": "fixed", "score": 0.3, "job_id": "ftjob-r"}],
    }
    report.write_text(json.dumps({"steps": [{"kind": "verdict"}, retrain, comparison]}))
    seeded = FixCache(storage, "seeded.json")
    assert seed_fixes(seeded, report) == 1
    assert seeded({**JOB_HP, "n_epochs": 1}) == 0.3


def test_reference_comes_from_the_b5_file(tmp_path):
    (tmp_path / "banking77.json").write_text(
        json.dumps({"Qwen/Qwen3-0.6B": {"heldout_exact_match": 0.45}})
    )
    assert reference_metric("banking77", "Qwen/Qwen3-0.6B", tmp_path) == 0.45
    assert reference_metric("banking77", "Qwen/Qwen3-1.7B", tmp_path) is None
    assert reference_metric("json", "Qwen/Qwen3-0.6B", tmp_path) is None


def test_cases_round_trip(tmp_path):
    case = _case(tmp_path, "healthy-s0", truth=())
    case.save(tmp_path / "cases")
    (loaded,) = load_cases("dev", tmp_path / "cases")
    assert loaded == case and loaded.healthy and loaded.train_examples == 2


def test_checklist_runs_on_every_case_and_is_scored(tmp_path):
    # target patch -> (target metrics, control metrics)
    pairs = {
        "learning_rate": ([0.5, 0.5], [0.12, 0.12]),
        "n_epochs": ([0.05, 0.06], [0.12, 0.12]),
        "lora_r": ([0.12, 0.13], [0.12, 0.12]),
    }

    def run_pair(target, control):
        target.metrics, control.metrics = pairs[next(iter(target.patch))]

    storage = LocalStorage(tmp_path)
    fixes = FixCache(storage, "fixes.json", lambda hp: (0.44, "ftjob-fix"))
    seen = []
    rows = run_case(
        _case(tmp_path),
        ["checklist"],
        router_for=lambda m: None,
        pairs=PairCache(storage, "pairs.json", run_pair),
        fixes=fixes,
        reference=REFERENCE,
        pair_cost_usd=0.06,
        on_diagnosis=seen.append,
    )
    (row,) = rows
    assert (row.method, row.cause, row.top1, row.fix_success) == (
        "checklist",
        "lr_too_low",
        True,
        True,
    )
    assert (row.mini_jobs, row.cost_usd) == (12, pytest.approx(0.18))
    assert seen[0].method == "checklist"

    # without training, a method that needs a pair that is not cached skips the case
    offline = PairCache(LocalStorage(tmp_path / "empty"), "pairs.json")
    skipped = run_case(
        _case(tmp_path),
        ["checklist"],
        router_for=lambda m: None,
        pairs=offline,
        fixes=fixes,
        reference=REFERENCE,
        pair_cost_usd=0.06,
    )
    assert skipped == []


def test_a_fix_with_the_reference_recipe_needs_no_training(tmp_path):
    reference_hp = {**JOB_HP, "learning_rate": 2e-4, "n_epochs": 6}
    (tmp_path / "banking77.json").write_text(
        json.dumps(
            {
                "Qwen/Qwen3-0.6B": {
                    "train_examples": 2,
                    "hyperparameters": reference_hp,
                    "heldout_exact_match": 0.535,
                    "job_id": "ftjob-ref",
                }
            }
        )
    )
    case = _case(tmp_path, learning_rate=2e-5, n_epochs=6)
    fixes = FixCache(LocalStorage(tmp_path), "fixes.json")
    assert seed_reference(fixes, case, tmp_path) == 1
    assert fixes(fixed_hyperparameters(case, "lr_too_low")) == 0.535

    (tmp_path / "other").mkdir()
    bigger = _case(tmp_path / "other", learning_rate=2e-5, n_epochs=6)
    (tmp_path / "other" / "train.jsonl").write_text('{"i": 1}\n' * 5)
    assert (
        seed_reference(FixCache(LocalStorage(tmp_path), "f2.json"), bigger, tmp_path)
        == 0
    )
