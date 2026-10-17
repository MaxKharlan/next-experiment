import argparse
from types import SimpleNamespace as NS

from core.connectors.storage import LocalStorage
from scripts.make_case import build_case, case_problem, sweep_cases

HP = {"n_epochs": 6, "learning_rate": 2e-5, "lora_r": 16, "lora_alpha": 16}


class FakeTF:
    def job(self, job_id):
        dump = {**HP, "lora": True, "packing": False}
        return NS(
            status="succeeded",
            model="Qwen/Qwen3-0.6B",
            hyperparameters=NS(model_dump=lambda exclude_none: dump),
        )

    def checkpoints(self, job_id):
        return [
            NS(id="ck-2", step_number=50, metrics=NS(train_loss=0.8, valid_loss=1.1)),
            NS(id="ck-1", step_number=25, metrics=NS(train_loss=1.53, valid_loss=1.6)),
        ]

    def download_checkpoint(self, checkpoint, storage, prefix):
        return [storage.put_text(f"{prefix}/{checkpoint.id}/adapter_config.json", "{}")]


def _args(tmp_path):
    for name in ("train", "valid", "heldout"):
        (tmp_path / f"{name}.jsonl").write_text('{"i": 1}\n{"i": 2}\n{"i": 3}\n')
    return argparse.Namespace(
        split="dev",
        task="banking77",
        train=str(tmp_path / "train.jsonl"),
        valid=str(tmp_path / "valid.jsonl"),
        heldout=str(tmp_path / "heldout.jsonl"),
        errors=1,
        report=[],
    )


def test_case_from_a_job_has_its_recipe_curve_metric_and_wrong_answers(tmp_path):
    def score(adapter, data, wrong):
        assert adapter.name == "ck-2"  # the last checkpoint
        wrong += [{"query": "q1", "expected": "a", "answer": "b"}] * 2
        return 1 / 3

    case = build_case(
        FakeTF(),
        LocalStorage(tmp_path / "artifacts"),
        score,
        "ftjob-1",
        "ref-fault6-mild-s0",
        ["lr_too_low"],
        _args(tmp_path),
    )
    assert case.hyperparameters == HP  # without lora and packing
    assert [c["train_loss"] for c in case.case["checkpoints"]] == [1.53, 0.8]
    assert case.case["symptom"].startswith(
        "After fine-tuning the model gives the right"
    )
    assert "33% of 3 held-out queries" in case.case["symptom"]
    assert len(case.case["wrong_answers"]) == 1
    assert (case.job_id, case.heldout_metric, case.truth) == (
        "ftjob-1",
        1 / 3,
        ["lr_too_low"],
    )


def test_sweep_runs_become_named_cases():
    rows = [
        {"run": "healthy-s0", "fault": None, "seed": 0, "status": "succeeded",
         "job_id": "j0"},
        {"run": "lr_too_low-mild-s0", "fault": "lr_too_low", "seed": 0,
         "status": "succeeded", "job_id": "j1"},
        {"run": "too_many_epochs-strong-s0", "fault": "too_many_epochs", "seed": 0,
         "status": "failed", "job_id": "j2"},
        {"run": "lora_rank_too_small-strong-s1", "fault": "lora_rank_too_small",
         "seed": 1, "status": "succeeded", "job_id": "j3"},
    ]  # fmt: skip
    assert sweep_cases(rows, "ref-") == [
        ("ref-healthy-s0", "j0", []),
        ("ref-fault6-mild-s0", "j1", ["lr_too_low"]),
        ("ref-fault8-strong-s1", "j3", ["lora_rank_too_small"]),
    ]


def test_cases_must_be_clean_on_either_side_of_the_reference():
    harmless = NS(healthy=False, heldout_metric=0.50)  # within the tolerance of 0.05
    assert case_problem(harmless, 0.535) == "the fault does not hurt on this task"
    assert case_problem(NS(healthy=False, heldout_metric=0.40), 0.535) is None
    assert case_problem(NS(healthy=True, heldout_metric=0.50), 0.535) is None
    weak = NS(healthy=True, heldout_metric=0.42)
    assert "not clean" in case_problem(weak, 0.535)
    assert case_problem(harmless, None) is None
