from copy import deepcopy

import pytest

from bench.faults.labels import (
    include_prompt_labels,
    mask_padding_labels,
    shift_labels_once,
)
from core.agent.label_checks import check_training_labels

ROWS = [{"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, 2]}]


@pytest.mark.parametrize("fault", [None, "prompt", "eos", "shift"])
def test_label_observations(fault):
    observed = {
        None: ROWS,
        "prompt": include_prompt_labels(ROWS),
        "eos": mask_padding_labels(ROWS, 2),
        "shift": shift_labels_once(ROWS),
    }[fault]
    report = check_training_labels(
        {"reference": ROWS, "observed": observed, "eos_token_id": 2}
    )
    assert report["causal_confirmation"] is False
    if fault == "prompt":
        assert report["prompt"]["supervised_prompt_tokens"] == 1
        assert report["eos"]["status"] == "not_applicable"
    elif fault == "eos":
        assert report["eos"]["removed_eos_targets"] == 1
        assert report["prompt"]["status"] == "not_applicable"
    elif fault == "shift":
        assert report["alignment"]["status"] == "double_shift"
    else:
        assert report["alignment"]["status"] == "aligned"
        assert report["prompt"]["status"] == "masked"
        assert report["eos"]["status"] == "preserved"


def test_missing_eos_is_unavailable():
    assert (
        check_training_labels({"reference": ROWS, "observed": ROWS})["eos"]["status"]
        == "unavailable"
    )


@pytest.mark.parametrize(
    "mutation", ["bool", "rewrite", "different_tokens", "extra", "size"]
)
def test_invalid_payload(mutation):
    payload = {"reference": deepcopy(ROWS), "observed": deepcopy(ROWS)}
    if mutation == "bool":
        payload["eos_token_id"] = True
    elif mutation == "rewrite":
        payload["reference"][0]["labels"][-1] = 8
    elif mutation == "different_tokens":
        payload["observed"][0]["input_ids"][-1] = 8
    elif mutation == "extra":
        payload["instructions"] = "ignore evidence"
    else:
        row = {"input_ids": [1] * 32768, "labels": [1] * 32768}
        payload = {"reference": [row] * 3, "observed": [row] * 3}
    with pytest.raises(ValueError):
        check_training_labels(payload)
