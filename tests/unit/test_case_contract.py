import pytest
from pydantic import ValidationError

from core.case import InvestigationCase


def _case():
    return {
        "task": "banking77",
        "symptom": "The fine-tuned model does not emit intent labels.",
        "job": {
            "id": "ftjob-1",
            "model": "Qwen/Qwen3-0.6B",
            "hyperparameters": {"learning_rate": 1e-5, "n_epochs": 3},
            "checkpoints": [{"epoch": 1, "train_loss": 3.25}],
        },
        "datasets": {
            "train": {"ref": "datasets/train.jsonl", "examples": 200},
            "validation": {"ref": "datasets/validation.jsonl", "examples": 50},
            "check": {"ref": "datasets/check.jsonl", "examples": 50},
            "test": {"ref": "datasets/test.jsonl", "examples": 200},
        },
        "evaluator": {
            "name": "intent_label",
            "metric": "exact_match",
        },
        "budget_usd": 1.0,
    }


def test_case_contract_parses_user_input():
    case = InvestigationCase.model_validate(_case())

    assert case.job.model == "Qwen/Qwen3-0.6B"
    assert case.datasets.test.examples == 200
    assert case.evaluator.higher_is_better is True
    assert case.mode == "A"


def test_all_dataset_splits_are_required():
    data = _case()
    del data["datasets"]["check"]

    with pytest.raises(ValidationError):
        InvestigationCase.model_validate(data)


def test_dataset_splits_must_use_different_files():
    data = _case()
    data["datasets"]["test"]["ref"] = data["datasets"]["check"]["ref"]

    with pytest.raises(ValidationError, match="must be different files"):
        InvestigationCase.model_validate(data)


@pytest.mark.parametrize("field", ["task", "symptom"])
def test_case_text_must_not_be_blank(field):
    data = _case()
    data[field] = "   "

    with pytest.raises(ValidationError):
        InvestigationCase.model_validate(data)


def test_budget_must_be_positive():
    data = _case()
    data["budget_usd"] = 0

    with pytest.raises(ValidationError):
        InvestigationCase.model_validate(data)


def test_unknown_fields_are_rejected():
    data = _case()
    data["dataset"] = "misspelled"

    with pytest.raises(ValidationError):
        InvestigationCase.model_validate(data)
