import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file

from core.agent.loop import investigate
from core.probes.level0.weights import LoraWeights, update_stats
from tests.unit.test_loop import FakeRouter, _ctx, _hyp, _runner

MODULES = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"]
SIGMAS = [3.0, 4.0, 0.0, 0.0]  # singular values of B @ A before scaling; r = 4


def _adapter(tmp_path, *, prefix="", alpha=8, rslora=False, sigmas=SIGMAS):
    """A rank-4 adapter whose updates have known singular values."""
    torch.manual_seed(0)
    r, out_dim, in_dim = len(sigmas), 12, 10
    tensors = {}
    for module in MODULES:
        u, _ = torch.linalg.qr(torch.randn(out_dim, r))
        v, _ = torch.linalg.qr(torch.randn(in_dim, r))
        tensors[f"{prefix}{module}.lora_B.weight"] = (
            u * torch.tensor(sigmas)
        ).contiguous()
        tensors[f"{prefix}{module}.lora_A.weight"] = v.T.contiguous()
    tmp_path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    config = {
        "r": r,
        "lora_alpha": alpha,
        "use_rslora": rslora,
        "base_model_name_or_path": "Qwen/Qwen3-0.6B",
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    return tmp_path


def _norms(value):
    return {f"{m}.weight": value for m in MODULES}


def test_update_norm_and_effective_rank_of_a_known_adapter(tmp_path):
    stats = update_stats(_adapter(tmp_path), _norms(1000.0))
    # |dW| = alpha / r * |(3, 4)| = 2 * 5
    assert stats.rel_median == pytest.approx(0.01, rel=1e-5)
    p = [3 / 7, 4 / 7]
    assert stats.erank_ratio == pytest.approx(
        math.exp(-sum(x * math.log(x) for x in p)) / 4, rel=1e-5
    )
    assert stats.top_share == pytest.approx(0.8, rel=1e-5)
    assert (stats.r, stats.modules, stats.model) == (4, 2, "Qwen/Qwen3-0.6B")


def test_peft_prefix_and_rslora_scaling(tmp_path):
    plain = update_stats(_adapter(tmp_path / "a"), _norms(1000.0))
    peft = update_stats(
        _adapter(tmp_path / "b", prefix="base_model.model."), _norms(1000.0)
    )
    rslora = update_stats(_adapter(tmp_path / "c", rslora=True), _norms(1000.0))
    assert peft.rel_median == pytest.approx(plain.rel_median)
    # alpha / sqrt(r) = 4 instead of alpha / r = 2
    assert rslora.rel_median == pytest.approx(2 * plain.rel_median)


def test_probe_needs_the_adapter_and_reads_small_updates(tmp_path):
    adapter = _adapter(tmp_path)
    assert not LoraWeights().available(_ctx())
    ctx = _ctx()
    ctx.files["adapter"] = str(adapter)
    assert LoraWeights().available(ctx)

    small = LoraWeights(norms=lambda model: _norms(1000.0)).run(ctx)
    assert small.outcome == "small"
    raw = json.loads(small.summary)
    assert (
        raw["rel_median"] == pytest.approx(0.01, rel=1e-3)
        and raw["small_below"] == LoraWeights.SMALL
    )
    assert LoraWeights(norms=lambda model: _norms(100.0)).run(ctx).outcome == "normal"


def test_the_loop_runs_the_weight_probe_only_when_the_adapter_is_there(tmp_path):
    probe = LoraWeights(norms=lambda model: _norms(1000.0))

    def probes_run(ctx):
        router = FakeRouter(
            [_hyp("h1", "lr_too_low", 0.4), _hyp("h2", "chat_template_mismatch", 0.3)]
        )
        run_pair, _ = _runner({"h1": ([0.16, 0.20], [0.02, 0.02])})
        result = investigate(ctx, {}, router, run_pair, probes=(probe,), explain=False)
        return [
            (s["probe"], s["outcome"]) for s in result.steps if s["kind"] == "probe"
        ]

    assert probes_run(_ctx()) == []
    ctx = _ctx()
    ctx.files["adapter"] = str(_adapter(tmp_path))
    assert probes_run(ctx) == [("lora_weights", "small")]
