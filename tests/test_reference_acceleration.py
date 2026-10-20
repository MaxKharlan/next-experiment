import json
from pathlib import Path

import pytest

from core.eval.runtime import file_digest
from jobs.cloud.reference_evaluation import choose, score_pairs
from scripts.accelerate_reference_evaluation import ready
from scripts.collect_reference_evaluation import analyze


def test_selects_every_validation_and_only_selected_check(tmp_path):
    plan = {
        "runtime": {"model": "Qwen/Qwen3-0.6B"},
        "tasks": {"banking77": {}, "function_calling": {}, "json_extraction": {}},
        "jobs": {},
        "state": {},
        "selection": "validation score, then grid index",
        "scope": "development",
    }
    for task in plan["tasks"]:
        for index in (3, 2, 1, 0):
            name = f"{task}-{index}"
            plan["jobs"][name] = {"task": task, "grid_index": index}
            plan["state"][name] = {"job_id": name}
    (tmp_path / "experiment.json").write_text(json.dumps(plan))
    calls = []

    def fake(root, output, protocol, snapshot, task, role, name, entry):
        calls.append((task, role, name))
        if role == "check" and entry:
            saved = json.loads(
                (output / "selection" / f"Qwen3-0.6B-{task}.json").read_text()
            )
            assert saved["selected"] == name
            return 0.8
        return 0.9 if name.endswith(("-1", "-2")) else 0.5

    score_pairs(tmp_path, tmp_path / "out", plan, Path("."), fake)
    result = json.loads((tmp_path / "out/results.json").read_text())
    assert result["status"] == "complete"
    assert len(calls) == 21
    for task in plan["tasks"]:
        checks = [n for t, role, n in calls if t == task and role == "check"]
        assert checks == [f"Qwen3-0.6B-{task}-base", f"{task}-1"]
        assert result["pairs"][f"Qwen3-0.6B-{task}"]["selected"] == f"{task}-1"


def test_incomplete_grid_cannot_select():
    with pytest.raises(ValueError):
        choose({"g0": 1}, {"g0": {"grid_index": 0}, "g1": {"grid_index": 1}})


def test_model_dispatch_requires_twelve_downloaded_successful_jobs():
    plan = {"jobs": {str(i): {"model": "model"} for i in range(12)}}
    state = {str(i): {"settled": True, "status": "succeeded"} for i in range(12)}
    assert ready(plan, state, "model")
    state["3"]["settled"] = False
    assert not ready(plan, state, "model")
    state["3"].update(settled=True, status="failed")
    assert not ready(plan, state, "model")
    del state["3"]
    assert not ready(plan, state, "model")


def test_collector_rejects_inflated_prediction_score(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    data = bundle / "check.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Where is my card?"},
                    {"role": "assistant", "content": "card_arrival"},
                ]
            }
        )
        + "\n"
    )
    plan = {
        "files_sha256": {"check.jsonl": file_digest(data)},
        "runtime": {"model": "Qwen/Qwen3-0.6B"},
        "original_protocol_sha256": "original",
        "tasks": {
            "banking77": {"check": {"path": "check.jsonl", "sha256": file_digest(data)}}
        },
        "jobs": {},
        "scope": "development",
    }
    (tmp_path / "protocol.json").write_text(json.dumps(plan))
    scores = tmp_path / "collected/evaluation/scores"
    scores.mkdir(parents=True)
    record = {
        "protocol_sha256": file_digest(tmp_path / "protocol.json"),
        "original_protocol_sha256": "original",
        "data_sha256": file_digest(data),
        "runtime": plan["runtime"],
        "device": "cuda",
        "job_id": None,
        "weights_sha256": None,
        "complete": False,
        "rows": [
            {
                "line": 1,
                "expected": "card_arrival",
                "source": None,
                "answer": "wrong_label",
                "score": 1.0,
                "hit_limit": False,
                "stopped_on_eos": True,
            }
        ],
    }
    (scores / "Qwen3-0.6B-banking77-base-check.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Prediction identity or score changed"):
        analyze(tmp_path, tmp_path / "collected")
