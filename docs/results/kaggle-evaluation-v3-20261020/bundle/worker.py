import os
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
(output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
result = {"examples": len(rows), "exact_match": sum(r["score"] for r in rows)/len(rows), "eos_share": sum(r["stopped_on_eos"] for r in rows)/len(rows), "limit_share": sum(r["hit_limit"] for r in rows)/len(rows), "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "packages": packages, "attention": model.config._attn_implementation, "precision": "IEEE float32; math SDPA", "protocol_sha256": hashlib.sha256((root / "experiment.json").read_bytes()).hexdigest()}
(output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
