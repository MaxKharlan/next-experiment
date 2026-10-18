import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from bench.ablations import fingerprint, frozen_path, run, selected_variants
from bench.cases import BenchCase
from bench.diagnosis import from_investigation
from core.agent.loop import investigate
from tests.unit.test_loop import FakeRouter, _ctx, _hyp, _runner


def test_no_causal_matches_full_prefix_and_preserves_ranking():
    hypotheses = [_hyp("h1", "lr_too_low", 0.4), _hyp("h2", "other", 0.3)]
    runner, _ = _runner({"h1": ([0.5, 0.52], [0.1, 0.12])})
    full = investigate(_ctx(), {}, FakeRouter(hypotheses), runner, explain=False)

    def forbidden(*args):
        pytest.fail("No pair building, cost estimation, training or retrain proposal")

    ctx = _ctx()
    result = investigate(
        ctx,
        {},
        FakeRouter(hypotheses),
        forbidden,
        explain=False,
        causal_checks=False,
        pair_factory=forbidden,
        estimate_pair_cost=forbidden,
        propose_retrain=forbidden,
    )
    index = next(
        i for i, step in enumerate(full.steps) if step["kind"] == "intervention"
    )
    assert result.steps[:-1] == full.steps[:index]
    assert result.steps[-1]["reason"] == "causal checks disabled for this ablation"
    diagnosis = from_investigation(result)
    assert diagnosis.top[0] == "lr_too_low"
    assert dict(diagnosis.ranking)["lr_too_low"] > 0.4
    assert diagnosis.status == "inconclusive"
    assert diagnosis.cause is None and diagnosis.confirmed == []
    assert diagnosis.patch == ""
    assert diagnosis.mini_jobs == diagnosis.training_cost_usd == 0
    assert result.retrain_plan is None
    assert ctx.budget_left_usd == 1.0


def test_high_prior_does_not_force_extra_probes_or_confirm():
    hypotheses = [_hyp("h1", "lr_too_low", 0.9)]

    def forbidden(*args):
        pytest.fail("Training is disabled")

    result = investigate(
        _ctx(), {}, FakeRouter(hypotheses), forbidden, causal_checks=False
    )
    assert [step["kind"] for step in result.steps] == ["hypotheses", "stop"]
    assert result.verdict is None
    assert result.status == "inconclusive"


def test_variant_selection():
    assert selected_variants([0, 1], ["no_causal"]) == [
        ("no_causal", "with_tavily", "no_causal", 0)
    ]
    assert len(selected_variants([0, 1])) == 6
    with pytest.raises(ValueError):
        selected_variants([], ["unknown"])


def test_selected_replay_needs_no_pair_cache_or_without_tavily(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXT_EXPERIMENT_WEIGHTS", "0")
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n" * 10)
    case = BenchCase(
        id="toy",
        split="dev",
        task="banking77",
        model="Qwen/Qwen3-0.6B",
        truth=["lr_too_low"],
        hyperparameters=_ctx().job_config["hyperparameters"],
        case={},
        train=str(train),
        valid="unused",
        heldout="unused",
    )
    args = SimpleNamespace(
        artifacts=tmp_path / "artifacts",
        priors=tmp_path / "priors",
        out=tmp_path / "out",
        random_seeds=[0],
        variants=["no_causal"],
        split="dev",
        noise_sd=0.019,
        min_effect=0.06,
        pairs_name="pairs-check",
    )
    path = frozen_path(args.priors, case, "with_tavily")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint(case),
                "hypotheses": [_hyp("h1", "lr_too_low", 0.9).model_dump()],
                "llm_calls": [],
            }
        )
    )
    run(args, [case])
    row = json.loads((args.out / "rows.jsonl").read_text())
    assert row["method"] == "no_causal" and row["top1"] is True
    assert row["status"] == "inconclusive"
    assert row["mini_jobs"] == 0
    assert row["fix_success"] is False
    assert not args.artifacts.exists()
    metadata = json.loads((args.out / "protocol.json").read_text())
    assert metadata["protocol_version"] == 2
    assert metadata["new_training_jobs"] == 0
    assert (args.out / "summary.md").is_file()
    assert asdict(case)["truth"] == ["lr_too_low"]
