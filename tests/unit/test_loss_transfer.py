import copy
import json
from pathlib import Path

import pytest

from bench.cases import load_cases
from bench.loss_transfer import evaluate


def model():
    return json.loads(
        Path("docs/results/loss-classifier-local-20261018/model.json").read_text()
    )


def test_frozen_transfer_reports_failures_without_refitting():
    frozen = model()
    before = copy.deepcopy(frozen)
    report = evaluate(load_cases("dev") + load_cases("heldout"), frozen)
    assert frozen == before
    assert report["refitted"] is False
    assert report["splits"]["heldout"]["accuracy"] == 0.75
    assert report["splits"]["heldout"]["balanced_accuracy_observed_classes"] == 0.5
    assert report["splits"]["heldout"]["confusion"]["healthy"]["lr_too_low"] == 2


def test_transfer_rejects_duplicate_jobs_and_bad_scaler():
    cases = load_cases("dev")
    cases[1].job_id = cases[0].job_id
    with pytest.raises(ValueError, match="unique Token Factory"):
        evaluate(cases, model())
    bad = model()
    bad["scale"][0] = 0
    with pytest.raises(ValueError, match="positive"):
        evaluate(load_cases("dev"), bad)
