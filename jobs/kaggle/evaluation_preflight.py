"""Check every bundled import and the GPU before downloading model weights."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def run(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "experiment.json").read_text())
    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append({"name": name, "ok": True, "detail": detail})
        except Exception as exc:  # noqa: BLE001 — aggregate independent dependency failures
            checks.append(
                {"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )

    def command(args):
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=120, check=False
        )
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-12000:])
        return result.stdout.strip()

    def version(name, expected):
        actual = importlib.metadata.version(name)
        if actual.split("+")[0] != expected:
            raise ValueError(f"{actual} != {expected}")
        return actual

    def digest(name, expected):
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
        return actual

    for name, expected in plan["files_sha256"].items():
        check(f"sha256:{name}", lambda n=name, e=expected: digest(n, e))
    for name, expected in plan["dependency_pins"].items():
        check(f"version:{name}", lambda n=name, e=expected: version(n, e))
    check(
        "pip_check",
        lambda: command(
            [
                os.environ["KAGGLE_BOOTSTRAP_PYTHON"],
                "-m",
                "pip",
                "--python",
                sys.executable,
                "check",
            ]
        ),
    )
    check(
        "nvidia_smi",
        lambda: command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    )
    # Includes imports inside loaders, which module import alone would miss.
    imports = {}
    for path in (root / "core/eval").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.setdefault(alias.name, set())
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imports.setdefault(node.module, set()).update(
                    alias.name for alias in node.names
                )
    for name in (
        "numpy",
        "jinja2",
        "tokenizers",
        "pydantic",
        "accelerate",
        "huggingface_hub",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.qwen2.tokenization_qwen2",
    ):
        imports.setdefault(name, set())
    for path in (root / "core/eval").glob("*.py"):
        imports.setdefault(f"core.eval.{path.stem}", set())

    def import_symbols(module, names):
        loaded = importlib.import_module(module)
        for name in names:
            getattr(loaded, name)
        return sorted(names)

    for module, names in sorted(imports.items()):
        check(
            f"import:{module}",
            lambda m=module, ns=names: import_symbols(m, ns),
        )

    def gpu():
        import torch
        from torch.nn import functional

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no CPU fallback")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
        assert torch.isfinite(x @ x.T).all().item()
        q = x.reshape(1, 1, 4, 4)
        assert (
            torch.isfinite(functional.scaled_dot_product_attention(q, q, q))
            .all()
            .item()
        )
        torch.cuda.synchronize()
        return {
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch_cuda": torch.version.cuda,
            "precision": "IEEE float32; math SDPA",
        }

    check("cuda_float32_matmul_and_math_sdpa", gpu)
    report = {
        "status": "passed" if all(c["ok"] for c in checks) else "failed",
        "python": sys.version,
        "platform": platform.platform(),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "pythonhome": os.environ.get("PYTHONHOME"),
        "checks": checks,
        "installed_packages": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
            )
        ),
    }
    (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"preflight": report["status"], "checks": checks}), flush=True)
    return report


if __name__ == "__main__":
    result = run(Path(__file__).parent, Path(os.environ["NEBIUS_OUTPUT_DIR"]))
    raise SystemExit(0 if result["status"] == "passed" else 1)
