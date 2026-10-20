"""Infrastructure failures should be reported together before model loading."""

import json
import sys
from types import SimpleNamespace

from jobs.kaggle import evaluation_preflight


def test_collects_missing_dependency_bad_digest_and_gpu_failure(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    source = root / "core/eval"
    source.mkdir(parents=True)
    (source / "generate.py").write_text(
        "def load():\n    from missing_loader_dependency import Model\n"
    )
    (root / "experiment.json").write_text(
        json.dumps(
            {
                "files_sha256": {"core/eval/generate.py": "wrong"},
                "dependency_pins": {"missing_distribution": "1.0"},
            }
        )
    )
    monkeypatch.setenv("KAGGLE_BOOTSTRAP_PYTHON", sys.executable)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    def import_module(name):
        if name == "missing_loader_dependency":
            raise ModuleNotFoundError(name)
        return SimpleNamespace()

    monkeypatch.setattr(evaluation_preflight.importlib, "import_module", import_module)
    monkeypatch.setattr(
        evaluation_preflight.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="simulated incompatible environment"
        ),
    )
    output = tmp_path / "output"
    report = evaluation_preflight.run(root, output)
    saved = json.loads((output / "preflight.json").read_text())
    failed = {c["name"] for c in saved["checks"] if not c["ok"]}
    assert report["status"] == saved["status"] == "failed"
    assert {
        "sha256:core/eval/generate.py",
        "version:missing_distribution",
        "import:missing_loader_dependency",
        "pip_check",
        "nvidia_smi",
        "cuda_float32_matmul_and_math_sdpa",
    } <= failed
    assert saved["checks"][-1]["name"] == "cuda_float32_matmul_and_math_sdpa"
