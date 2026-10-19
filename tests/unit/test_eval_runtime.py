import hashlib
import json
from pathlib import Path

import pytest

from apps.api.live_protocol import verify_datasets
from core.eval import evaluate, generate
from core.eval.runtime import REQUIRED_FILES, freeze_runtime, verify_runtime
from tests.unit.test_eval import SPEC, FakeModel, FakeTokenizer, _write


def frozen(tmp_path, monkeypatch):
    root = tmp_path / ("a" * 40)
    root.mkdir()
    for name in REQUIRED_FILES:
        (root / name).write_text("fixture")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "fixed template"})
    )
    monkeypatch.setattr(
        "core.eval.runtime.importlib.metadata.version", lambda name: "1.0"
    )
    return freeze_runtime(root, "Qwen/Qwen3-0.6B")


def test_runtime_detects_weight_changes_before_loading(tmp_path, monkeypatch):
    runtime = frozen(tmp_path, monkeypatch)
    data = _write(tmp_path / "check.jsonl", [("question", "answer")])
    (Path(runtime["snapshot"]) / "model.safetensors").write_text("changed weights")
    monkeypatch.setattr(generate, "load_model", lambda *a: pytest.fail("loaded model"))
    with pytest.raises(ValueError, match="snapshot changed"):
        evaluate(data, SPEC, runtime=runtime)


def test_runtime_rejects_dependency_and_template_changes(tmp_path, monkeypatch):
    runtime = frozen(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.eval.runtime.importlib.metadata.version", lambda name: "2.0"
    )
    with pytest.raises(ValueError, match="dependency changed"):
        verify_runtime(runtime)
    monkeypatch.setattr(
        "core.eval.runtime.importlib.metadata.version", lambda name: "1.0"
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "chat_template.jinja").write_text("different template")
    with pytest.raises(ValueError, match="chat template"):
        verify_runtime(runtime, template_dir=adapter)
    with pytest.raises(ValueError, match="model differs"):
        verify_runtime(runtime, model="other/model")


def test_pinned_evaluation_loads_local_snapshot_and_records_binding(
    tmp_path, monkeypatch
):
    runtime = frozen(tmp_path, monkeypatch)
    data = _write(tmp_path / "check.jsonl", [("question", "answer")])
    tok = FakeTokenizer()
    model = FakeModel(tok, ["answer"])
    seen = []

    def load(adapter, base, template):
        seen.append(base)
        return model, tok, base

    monkeypatch.setattr(generate, "load_model", load)
    result = evaluate(data, SPEC, runtime=runtime, device="cpu")
    assert seen == [runtime["snapshot"]]
    assert result.model == runtime["model"]
    assert result.runtime == runtime
    assert result.value == 1


def test_runtime_changes_during_generation_reject_the_score(tmp_path, monkeypatch):
    runtime = frozen(tmp_path, monkeypatch)
    data = _write(tmp_path / "check.jsonl", [("question", "answer")])
    tok = FakeTokenizer()
    model = FakeModel(tok, ["answer"])
    monkeypatch.setattr(generate, "load_model", lambda *a: (model, tok, "model"))
    original = generate.answers

    def changed(*args):
        rows = original(*args)
        (Path(runtime["snapshot"]) / "config.json").write_text("changed")
        return rows

    monkeypatch.setattr(generate, "answers", changed)
    with pytest.raises(ValueError, match="snapshot changed"):
        evaluate(data, SPEC, runtime=runtime, device="cpu")


def test_live_dataset_gate_also_checks_runtime(tmp_path, monkeypatch):
    runtime = frozen(tmp_path, monkeypatch)
    data = _write(tmp_path / "check.jsonl", [("q", "a")])
    protocol = {
        "model": runtime["model"],
        "evaluation_runtime": runtime,
        "datasets": {
            "check": {
                "path": str(data),
                "examples": 1,
                "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            }
        },
    }
    verify_datasets(protocol)
    (Path(runtime["snapshot"]) / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="snapshot changed"):
        verify_datasets(protocol)


def test_dataset_change_during_generation_rejects_the_score(tmp_path, monkeypatch):
    runtime = frozen(tmp_path, monkeypatch)
    data = _write(tmp_path / "check.jsonl", [("question", "answer")])
    tok = FakeTokenizer()
    model = FakeModel(tok, ["answer"])
    monkeypatch.setattr(generate, "load_model", lambda *a: (model, tok, "model"))
    original = generate.answers

    def changed(*args):
        rows = original(*args)
        data.write_text("changed")
        return rows

    monkeypatch.setattr(generate, "answers", changed)
    with pytest.raises(ValueError, match="dataset changed"):
        evaluate(data, SPEC, runtime=runtime, device="cpu")
