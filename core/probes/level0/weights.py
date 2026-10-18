"""B25: a level-0 probe on the weights of the job's LoRA adapter. No inference, no LLM.

Token Factory returns a PEFT adapter in result_files, not merged weights
(docs/b6-post-training.md), so each adapted matrix changes by dW = s * B @ A with
s = lora_alpha / r (lora_alpha / sqrt(r) with rsLoRA). For every adapted matrix the
probe measures:

- the Frobenius norm of dW divided by the norm of the base weight it changes;
- the singular values of dW, from the r x r core of the QR factors of B and A^T;
- their effective rank (Roy and Vetterli, 2007), exp of the entropy of the
  normalised singular values, divided by r.

A job whose learning rate was too low barely moves its weights, so the median
relative norm is small: that is the "small" outcome and the evidence for fault 6.
Effective rank close to r with a flat spectrum was meant to point at fault 8, but on
Token Factory healthy rank-16 adapters already use about 0.9 of their rank, so the
probe reports it and does not turn it into an outcome. docs/b25-lora-weights.md has
the measurements behind both decisions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.probes.base import Probe, ProbeResult

if TYPE_CHECKING:
    from core.agent.context import InvestigationContext

PEFT_PREFIX = "base_model.model."


@dataclass
class WeightStats:
    model: str
    r: int
    alpha: float
    modules: int
    rel_median: float  # median over modules of |dW| / |W|
    rel_max: float
    erank_ratio: float  # mean over modules of effective rank / r
    top_share: float  # mean share of the largest singular value in |dW|


def read_adapter(adapter_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The adapter's config and its tensors, keyed without the PEFT prefix."""
    from safetensors import safe_open

    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    tensors = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open is not a dict
            tensors[key.removeprefix(PEFT_PREFIX)] = f.get_tensor(key).float()
    return config, tensors


def update_stats(adapter_dir: Path, base_norms: Mapping[str, float]) -> WeightStats:
    """Relative norm and spectrum of every update the adapter makes."""
    import torch

    config, tensors = read_adapter(adapter_dir)
    r, alpha = int(config["r"]), float(config["lora_alpha"])
    scale = alpha / math.sqrt(r) if config.get("use_rslora") else alpha / r
    rel, erank, top = [], [], []
    for key, a in tensors.items():
        if ".lora_A." not in key:
            continue
        b = tensors[key.replace(".lora_A.", ".lora_B.")]
        _, rb = torch.linalg.qr(b)
        _, ra = torch.linalg.qr(a.T)
        s = torch.linalg.svdvals(rb @ ra.T) * scale
        norm = float(s.norm())
        rel.append(norm / base_norms[key.replace(".lora_A.weight", ".weight")])
        if norm == 0:
            erank.append(0.0)
            top.append(0.0)
            continue
        p = s / s.sum()
        p = p[p > 0]
        erank.append(math.exp(-float((p * p.log()).sum())) / r)
        top.append(float(s[0]) / norm)
    if not rel:
        raise ValueError(f"{adapter_dir}: no LoRA matrices")
    rel_sorted = sorted(rel)
    return WeightStats(
        model=config.get("base_model_name_or_path", ""),
        r=r,
        alpha=alpha,
        modules=len(rel),
        rel_median=rel_sorted[len(rel) // 2],
        rel_max=rel_sorted[-1],
        erank_ratio=sum(erank) / len(erank),
        top_share=sum(top) / len(top),
    )


@lru_cache(maxsize=4)
def base_norms(model: str, cache_dir: str = "artifacts/probes") -> dict[str, float]:
    """Frobenius norm of every 2-D weight of the base model, cached as JSON.

    The first call downloads the model's safetensors from the Hugging Face Hub if
    they are not in the local cache and reads them one tensor at a time.
    """
    cache = Path(cache_dir) / f"base-norms-{model.replace('/', '--')}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(snapshot_download(model, allow_patterns=["*.safetensors"]))
    norms = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), "pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open is not a dict
                if key.endswith(".weight") and len(f.get_slice(key).get_shape()) == 2:
                    norms[key] = float(f.get_tensor(key).float().norm())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(norms))
    return norms


class LoraWeights(Probe):
    """Did training move the adapter's weights enough to change the model?

    Needs the job's adapter in ctx.files["adapter"]: a local directory with
    adapter_config.json and adapter_model.safetensors.
    """

    name = "lora_weights"
    level = 0
    outcomes = ("small", "normal")
    # median relative update norm, set on the 4 dev cases by
    # scripts/calibrate_lora_weights.py (docs/b25-lora-weights.md)
    SMALL = 0.0195

    def __init__(self, norms: Callable[[str], Mapping[str, float]] = base_norms):
        self.norms = norms

    def available(self, ctx: InvestigationContext) -> bool:
        adapter = ctx.files.get("adapter")
        return bool(adapter) and (Path(adapter) / "adapter_model.safetensors").exists()

    def estimate_cost(self, ctx: InvestigationContext) -> float:
        return 0.0

    def run(self, ctx: InvestigationContext) -> ProbeResult:
        adapter = Path(ctx.files["adapter"])
        config = json.loads((adapter / "adapter_config.json").read_text())
        stats = update_stats(adapter, self.norms(config["base_model_name_or_path"]))
        outcome = "small" if stats.rel_median < self.SMALL else "normal"
        raw = {
            **{
                k: round(v, 5) if isinstance(v, float) else v
                for k, v in asdict(stats).items()
            },
            "small_below": self.SMALL,
        }
        return self.check_outcome(
            ProbeResult(
                outcome=outcome,
                raw_ref=f"investigations/{ctx.investigation_id}/probes/{self.name}.json",
                summary=json.dumps(raw),
            )
        )
