"""Package a credential-free, frozen 64-example CUDA evaluation notebook."""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from core.eval.runtime import file_digest

WORKER = """import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from core.eval.data import read_examples
from core.eval.generate import answer, load_model
from core.eval.metrics import exact_match

root = Path(__file__).parent
plan = json.loads((root / "experiment.json").read_text())
for name, digest in plan["files_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
assert torch.cuda.is_available(), "A CUDA GPU is required; no CPU fallback"
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
packages = {name: importlib.metadata.version(name) for name in plan["runtime"]["packages"]}
for name, value in packages.items():
    assert value.split("+")[0] == plan["runtime"]["packages"][name]
snapshot = Path(snapshot_download(plan["runtime"]["model"], revision=plan["runtime"]["revision"]))
for name, digest in plan["runtime"]["files_sha256"].items():
    assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == digest
model, tokenizer, _ = load_model(None, base_model=str(snapshot))
model.to("cuda").eval()
rows = []
for example in read_examples(root / "check.jsonl"):
    with torch.inference_mode():
        row = asdict(answer(model, tokenizer, example, "cuda", exact_match, 32))
    if row["hit_limit"]:
        row["score"] = 0.0
    rows.append(row)
output = Path(os.environ["NEBIUS_OUTPUT_DIR"])
output.mkdir(parents=True, exist_ok=True)
(output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\\n")
result = {"examples": len(rows), "exact_match": sum(r["score"] for r in rows)/len(rows), "eos_share": sum(r["stopped_on_eos"] for r in rows)/len(rows), "limit_share": sum(r["hit_limit"] for r in rows)/len(rows), "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "packages": packages, "attention": model.config._attn_implementation, "precision": "IEEE float32; math SDPA", "protocol_sha256": hashlib.sha256((root / "experiment.json").read_bytes()).hexdigest()}
(output / "result.json").write_text(json.dumps(result, indent=2) + "\\n")
print(json.dumps(result))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--without-ensurepip", action="store_true")
    args = parser.parse_args()
    destination = args.destination
    bundle = destination / "bundle"
    bundle.mkdir(parents=True, exist_ok=False)
    source = Path("docs/results/pretrained-code-audit-20261020")
    runtime = json.loads((source / "protocol.json").read_text())["runtime"]
    for name in ("data.py", "generate.py", "metrics.py", "records.py", "structured.py"):
        target = bundle / "core/eval" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path("core/eval") / name, target)
    shutil.copyfile(source / "check.jsonl", bundle / "check.jsonl")
    shutil.copyfile(
        "docs/results/cloud-parity-isolated-20261020/bundle/main.py", bundle / "main.py"
    )
    if args.without_ensurepip:
        bootstrap = bundle / "main.py"
        bootstrap.write_text(
            bootstrap.read_text()
            .replace("import subprocess\n", "import subprocess\nimport sys\n")
            .replace("with_pip=True", "with_pip=False")
            .replace(
                '[python, "-m", "pip", "install",',
                '[sys.executable, "-m", "pip", "--python", python, "install",',
            )
        )
    (bundle / "worker.py").write_text(WORKER)
    (bundle / "dependencies.txt").write_text(
        "".join(f"{name}=={value}\n" for name, value in runtime["packages"].items())
    )
    plan = {
        "id": destination.name,
        "bootstrap_amendment": (
            "The predecessor failed in ensurepip before dependency installation or model loading. Use the notebook's pip with --python to install into an empty isolated venv; all model, data, generation and package pins are unchanged."
            if args.without_ensurepip
            else None
        ),
        "runtime": runtime,
        "model_checkpoint": "Pinned base model, no adapter",
        "generation": {"greedy": True, "thinking": False, "max_new_tokens": 32},
        "scope": "Prospective Kaggle comparison against existing MPS 64-example predictions; development data, no training or final test.",
        "attempts": 1,
        "stopping": "Record every disagreement. No threshold changes or automatic retry after generation starts.",
        "comparison": "Exact prediction and metric differences versus MPS and a separately recorded CUDA cohort; no claim of training backend equivalence.",
        "export": "Download kernel outputs through Kaggle API, then upload locally to existing Object Storage under kaggle/kaggle-base-generation-20261020/. No S3 keys in Kaggle.",
        "files_sha256": {
            str(p.relative_to(bundle)): file_digest(p)
            for p in bundle.rglob("*")
            if p.is_file()
        },
    }
    (bundle / "experiment.json").write_text(json.dumps(plan, indent=2) + "\n")
    (destination / "protocol.json").write_text(json.dumps(plan, indent=2) + "\n")
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.writestr(path.relative_to(bundle).as_posix(), path.read_bytes())
    payload = base64.b64encode(data.getvalue()).decode()
    code = f"""import base64, hashlib, io, os, pathlib, subprocess, sys, zipfile
raw = base64.b64decode({payload!r})
assert hashlib.sha256(raw).hexdigest() == {__import__("hashlib").sha256(data.getvalue()).hexdigest()!r}
root = pathlib.Path('/kaggle/working/frozen-eval')
root.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for item in archive.infolist():
        p = pathlib.Path(item.filename)
        assert not p.is_absolute() and '..' not in p.parts
    archive.extractall(root)
os.environ['NEBIUS_OUTPUT_DIR'] = '/kaggle/working/evaluation'
subprocess.run([sys.executable, str(root / 'main.py')], check=True)
"""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "scope",
                "metadata": {},
                "source": [
                    "# Frozen CUDA evaluation\n",
                    "Requires GPU and Internet. One evaluation attempt; no training. Package versions, model files and inputs are checked before generation. Download evaluation/result.json and evaluation/predictions.json after completion.",
                ],
            },
            {
                "cell_type": "code",
                "id": "evaluate",
                "metadata": {},
                "source": code.splitlines(keepends=True),
                "execution_count": None,
                "outputs": [],
            },
        ],
    }
    path = args.notebook
    path.write_text(json.dumps(notebook, indent=2) + "\n")
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--fix", str(path)], check=True
    )
    subprocess.run([sys.executable, "-m", "ruff", "format", str(path)], check=True)
    (destination / "preparation.json").write_text(
        json.dumps(
            {
                "notebook_sha256": file_digest(path),
                "protocol_sha256": file_digest(destination / "protocol.json"),
                "payload_bytes": len(data.getvalue()),
                "status": "prepared_not_executed",
                "credentials": "Account credential is used only by the local Kaggle client; no credential enters this notebook",
            },
            indent=2,
        )
        + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
