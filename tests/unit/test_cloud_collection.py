import copy

import pytest

from core.eval.data import Example
from core.eval.runtime import file_digest
from scripts.collect_cloud_matrix import analyze, safe_relative, verify_predictions


def test_prediction_verification_rejects_identity_and_score_changes():
    examples = [Example(1, [], "card_arrival", "card_arrival")]
    rows = [
        {
            "line": 1,
            "expected": "card_arrival",
            "source": "card_arrival",
            "answer": "card_arrival",
            "score": 1.0,
            "hit_limit": False,
            "stopped_on_eos": True,
        }
    ]
    assert verify_predictions(rows, examples)["exact_match"] == 1
    for field, value in (("line", 2), ("score", 0), ("hit_limit", True)):
        corrupted = copy.deepcopy(rows)
        corrupted[0][field] = value
        with pytest.raises(ValueError):
            verify_predictions(corrupted, examples)
    with pytest.raises(ValueError):
        verify_predictions([], examples)


def test_missing_cells_cannot_be_reported_as_complete(tmp_path):
    protocol = tmp_path / "experiment.json"
    protocol.write_text("{}")
    plan = {
        "id": "matrix",
        "runtime": {"revision": "model-version", "packages": {"torch": "1"}},
        "arms": [{"name": "healthy"}],
        "seeds": [1, 2, 3],
    }
    result = {
        "id": "matrix",
        "status": "complete",
        "cells": {},
        "runtime": {
            "model_revision": "model-version",
            "protocol_sha256": file_digest(protocol),
            "packages": {"torch": "1+cu"},
            "precision": {
                "matmul": "highest",
                "tf32": False,
                "NVIDIA_TF32_OVERRIDE": "0",
                "sdpa": "math only",
            },
        },
    }
    with pytest.raises(ValueError, match="missing cells"):
        analyze(plan, result, tmp_path, tmp_path)


def test_artifact_paths_cannot_escape_destination():
    for path in ("../weights", "/weights", "cell/../../weights", ""):
        with pytest.raises(ValueError):
            safe_relative(path)
    assert safe_relative("cell/adapter/weights").as_posix() == "cell/adapter/weights"
