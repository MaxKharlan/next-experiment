"""One prospective CUDA comparison against a completed MPS development audit."""

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.eval.data import read_examples
from core.probes.level1.logits import aggregate, encode_regions, inspect_example


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def main():
    started = time.time()
    root = Path(__file__).parent
    local = json.loads((root / "local-protocol.json").read_text())
    assert torch.cuda.is_available(), "CUDA is required; no CPU fallback"
    packages = {
        name: importlib.metadata.version(name)
        for name in local["runtime"]["packages"]
    }
    for name, expected in local["runtime"]["packages"].items():
        assert packages[name].split("+")[0] == expected, (name, packages[name])
    snapshot = Path(
        snapshot_download(
            local["runtime"]["model"], revision=local["runtime"]["revision"]
        )
    )
    for name, expected in local["runtime"]["files_sha256"].items():
        assert sha(snapshot / name) == expected, name
    data = root / "development.jsonl"
    assert sha(data) == local["data_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32)
    model.to("cuda").eval()
    rows = []
    token_hashes = []
    for example in read_examples(data):
        regions = encode_regions(tokenizer, example, context_length=8192)
        token_hashes.append(
            hashlib.sha256(
                json.dumps(regions.input_ids, separators=(",", ":")).encode()
            ).hexdigest()
        )
        rows.append(
            inspect_example(model, tokenizer, example, "cuda", context_length=8192)
        )
        print("completed", len(rows), flush=True)
    result = {
        "id": "cloud-parity-isolated-20261020",
        "rows": rows,
        "summary": aggregate(rows),
        "token_hashes": token_hashes,
        "packages": packages,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "attention": model.config._attn_implementation,
        "dtype": str(model.dtype),
        "model_revision": local["runtime"]["revision"],
        "data_sha256": sha(data),
        "wall_seconds": time.time() - started,
    }
    output = Path(os.environ["NEBIUS_OUTPUT_DIR"])
    output.joinpath("result.json").write_text(json.dumps(result, indent=2) + "\n")
    # No credentials or source text in logs. Results are retrieved through Job logs.
    print("PARITY_RESULT=" + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
