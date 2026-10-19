import json

import pytest

from core.report.kaggle import analyze
from jobs.kaggle.export_answer_cut import bundle

HP = {"n_epochs": 1, "batch_size": 8, "learning_rate": 1e-4}
METRICS = {
    "healthy": 0.88,
    "answers-cut-30": 0.66,
    "cut30-target": 0.80,
    "cut30-control": 0.62,
}


def _write_rows(path, n, tag):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"i": i, "tag": tag}) + "\n" for i in range(n)))


def _output(root, model="Qwen/Qwen3-0.6B"):
    tasks = root / "artifacts" / "tasks"
    files = {
        "healthy": (tasks / "banking77" / "train.jsonl", 10),
        "answers-cut-30": (tasks / "banking77-answers-cut-30" / "train.jsonl", 10),
        "cut30-target": (root / "runs" / "target-train.jsonl", 7),
        "cut30-control": (root / "runs" / "control-train.jsonl", 7),
    }
    _write_rows(tasks / "banking77" / "check.jsonl", 5, "check")
    effect, pair = {}, {}
    for name, (path, n) in files.items():
        _write_rows(path, n, name)
        for seed in (0, 1):
            key = f"{name}-s{seed}"
            run = root / "runs" / key
            run.mkdir(parents=True)
            summary = {
                "model": model,
                "seed": seed,
                "hyperparameters": HP,
                "train_examples": n,
                "final": {"valid_loss": 0.1},
            }
            (run / "summary.json").write_text(json.dumps(summary))
            row = {"exact_match": METRICS[name] + seed / 100}
            (pair if name.startswith("cut30") else effect)[key] = row
    (root / "runs" / "results.json").write_text(json.dumps(effect))
    (root / "runs" / "pair.json").write_text(json.dumps(pair))


def test_bundle_has_every_role_on_check_and_passes_the_analyzer(tmp_path):
    _output(tmp_path)
    data = bundle(tmp_path, 30, {"gpu": "T4"})
    assert data["evaluation"]["split"] == "check"
    assert data["evaluation"]["examples"] == 5
    roles = sorted({r["role"] for r in data["runs"]})
    assert roles == ["control", "damaged", "healthy", "target"]
    assert all(r["hyperparameters"] == HP for r in data["runs"])
    assert all("summary_path" not in r for r in data["runs"])
    assert data["control_description"]

    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(data))
    report = analyze(path, min_effect=0.06)
    assert report["status"] == "confirmed"
    assert report["quality_drop"] == pytest.approx(0.22)


def test_a_strength_without_a_pair_has_no_control(tmp_path):
    _output(tmp_path)
    tasks = tmp_path / "artifacts" / "tasks"
    _write_rows(tasks / "banking77-answers-cut-10" / "train.jsonl", 10, "ten")
    data = bundle(tmp_path, 10, {})
    assert sorted({r["role"] for r in data["runs"]}) == ["healthy"]
    assert "control_description" not in data


def test_a_summary_for_another_model_is_refused(tmp_path):
    _output(tmp_path, model="other/model")
    with pytest.raises(ValueError, match="not Qwen"):
        bundle(tmp_path, 30, {})
