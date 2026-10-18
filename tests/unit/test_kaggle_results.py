import copy
import json
import subprocess
import sys

import pytest

from core.report.kaggle import analyze


@pytest.fixture
def bundle(tmp_path):
    data = {
        "version": 1,
        "backend": "kaggle",
        "model": "Qwen/Qwen3-0.6B",
        "fault": "answer_truncation",
        "evaluation": {"split": "check", "sha256": "a" * 64, "examples": 200},
        "control_description": "Drop the same number of random unflagged rows",
        "runs": [
            {
                "run_id": f"{role}-{seed}",
                "role": role,
                "seed": seed,
                "metric": score,
                "train_sha256": sha * 64,
                "hyperparameters": {"n_epochs": 3, "learning_rate": 0.0002},
            }
            for role, sha, scores in [
                ("target", "b", [0.50, 0.52]),
                ("control", "c", [0.10, 0.12]),
            ]
            for seed, score in enumerate(scores)
        ],
    }
    path = tmp_path / "results.json"

    def save(value=data):
        path.write_text(json.dumps(value))
        return path

    return data, save


def test_causal_pair_and_original_file_unchanged(bundle):
    _, save = bundle
    path = save()
    original = path.read_bytes()
    report = analyze(path, min_effect=0.06)
    assert report["status"] == "confirmed"
    assert report["causal_result"]["effect"] == pytest.approx(0.4)
    assert report["noise_sd"] is None
    assert report["permutation_can_reach_0_05"] is False
    assert report["new_training_jobs"] == report["new_api_calls"] == 0
    assert path.read_bytes() == original


def test_healthy_vs_damaged_is_not_a_causal_pair(bundle):
    data, save = bundle
    for run in data["runs"]:
        run["role"] = {"target": "healthy", "control": "damaged"}[run["role"]]
    report = analyze(save(), min_effect=0.06)
    assert report["status"] == "insufficient_evidence"
    assert report["quality_drop"] == pytest.approx(0.4)
    assert report["causal_result"] is None


@pytest.mark.parametrize("split", ["validation", "test"])
def test_non_check_metrics_are_descriptive_only(bundle, split):
    data, save = bundle
    data["evaluation"]["split"] = split
    assert analyze(save(), min_effect=0.06)["status"] == "insufficient_evidence"


@pytest.mark.parametrize(
    "problem", ["nan", "duplicate", "seed", "hp", "same_data", "control"]
)
def test_invalid_pairs_are_rejected(bundle, problem):
    data, save = bundle
    if problem == "nan":
        data["runs"][0]["metric"] = float("nan")
    elif problem == "duplicate":
        data["runs"].append(copy.deepcopy(data["runs"][0]))
    elif problem == "seed":
        data["runs"][-1]["seed"] = 9
    elif problem == "hp":
        data["runs"][-1]["hyperparameters"]["n_epochs"] = 4
    elif problem == "same_data":
        data["runs"][-1]["train_sha256"] = "b" * 64
    else:
        data["control_description"] = ""
    with pytest.raises(ValueError):
        analyze(save(), min_effect=0.06)


def test_missing_seed_does_not_confirm(bundle):
    data, save = bundle
    data["runs"] = [run for run in data["runs"] if run["seed"] == 0]
    assert analyze(save(), min_effect=0.06)["status"] == "insufficient_evidence"


def test_legacy_summary_cannot_be_relabeled_check(bundle):
    data, save = bundle
    path = save()
    run = data["runs"][0]
    summary = {
        "model": data["model"],
        "seed": run["seed"],
        "hyperparameters": run["hyperparameters"],
        "valid_examples": 200,
        "final": {"exact_match": run["metric"]},
    }
    (path.parent / "summary.json").write_text(json.dumps(summary))
    run["summary_path"] = "summary.json"
    with pytest.raises(ValueError, match="Legacy summary"):
        analyze(save(), min_effect=0.06)
    data["evaluation"]["split"] = "validation"
    report = analyze(save(), min_effect=0.06)
    assert report["source_files"][0]["sha256"]
    assert report["status"] == "insufficient_evidence"


def test_cli_offline_and_refuses_overwrite(bundle):
    _, save = bundle
    path = save()
    out = path.parent / "report.json"
    code = """
import runpy, sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'openai', 'httpx'}:
            raise AssertionError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
runpy.run_module('scripts.analyze_kaggle', run_name='__main__')
"""
    command = [
        sys.executable,
        "-c",
        code,
        str(path),
        "--min-effect",
        "0.06",
        "--out",
        str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    before = out.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert out.read_bytes() == before
