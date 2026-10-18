import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from bench.ablations import fingerprint, frozen_path
from bench.pair_cache import PairCache, PairNotCached
from bench.routing_ablation import run
from core.agent.router import STEP_ROLES, Router
from core.agent.schemas import ProbeInterpretation, Verdict
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import CallResult, StructuredOutputError
from core.interventions.pairs import make_pair
from core.telemetry.calls import CallLog
from tests.unit.test_bench import _case
from tests.unit.test_loop import _ctx, _hyp


class NarrativeProvider:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def model(self, role):
        return f"test-{role}"

    def structured(self, model, messages, schema, **kwargs):
        self.calls.append((model, schema))
        call = CallResult(model, "{}", 100, 20, 0.002, 10)
        if self.fail:
            raise StructuredOutputError("Invalid saved response", [call])
        if schema is Verdict:
            parsed = Verdict(
                hypothesis_id="",
                status="rejected",
                evidence=[],
                patch="",
                summary=f"Explanation from {model}",
            )
        else:
            assert schema is ProbeInterpretation
            parsed = ProbeInterpretation(
                probe_name="", outcome="", explanation="Saved explanation"
            )
        return parsed, [call]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXT_EXPERIMENT_WEIGHTS", "0")
    case = _case(tmp_path)
    args = SimpleNamespace(
        allow_api_calls=True,
        split="dev",
        artifacts=tmp_path / "artifacts",
        priors=tmp_path / "priors",
        out=tmp_path / "out",
        pairs_name="pairs-check",
        min_effect=0.06,
        noise_sd=0.019,
    )
    hyp = _hyp("h1", "lr_too_low", 0.4)
    log = CallLog("prepared")
    log.add(
        "hypotheses", "hypotheses-v3", CallResult("test-super", "{}", 100, 10, 0.001, 5)
    )
    path = frozen_path(args.priors, case, "with_tavily")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint(case),
                "hypotheses": [hyp.model_dump()],
                "llm_calls": [asdict(c) for c in log.calls],
            }
        )
    )
    target, control = make_pair(hyp, _ctx())
    target.metrics = [0.5, 0.52]
    control.metrics = [0.1, 0.12]
    cache = PairCache(
        LocalStorage(args.artifacts), f"bench/dev/{case.id}/pairs-check.json"
    )
    cache.add(target, control)
    return args, case, path


def test_super_only_overrides_all_steps_and_escalation():
    router = Router(NarrativeProvider(), CallLog("test"), super_only=True)
    for _ in range(3):
        router.mismatch()
    assert {router.role(step) for step in STEP_ROLES} == {"super"}
    with pytest.raises(KeyError):
        router.role("unknown")


def test_routing_comparison_uses_cached_training_and_records_real_models(experiment):
    args, case, _ = experiment
    before = {p: p.read_bytes() for p in args.artifacts.rglob("*.json")}
    tf = NarrativeProvider()
    run(args, [case], tf=tf)
    assert ("test-ultra", Verdict) in tf.calls
    assert ("test-super", Verdict) in tf.calls
    assert ("test-super", ProbeInterpretation) in tf.calls
    rows = json.loads((args.out / "rows.json").read_text())
    assert [r["variant"] for r in rows] == ["mixed", "super_only"]
    for row in rows:
        assert row["status"] == "confirmed" and row["cause"] == "lr_too_low"
        assert row["cached_training_jobs"] == 4
        assert row["llm_cost_usd"] == pytest.approx(row["new_llm_cost_usd"] + 0.001)
        record = json.loads(
            (args.out / f"{case.id}--{row['variant']}.calls.json").read_text()
        )
        assert len(record["shared_hypothesis_calls"]) == 1
        assert len(record["new_calls"]) == row["new_llm_calls"]
        if row["variant"] == "super_only":
            assert {c["model"] for c in record["new_calls"]} == {"test-super"}
    assert before == {p: p.read_bytes() for p in args.artifacts.rglob("*.json")}
    with pytest.raises(FileExistsError):
        run(args, [case], tf=tf)


@pytest.mark.parametrize(
    "problem", ["authorization", "stale", "missing-pair", "wrong-model"]
)
def test_invalid_inputs_fail_before_model_calls(experiment, problem):
    args, case, path = experiment
    if problem == "authorization":
        args.allow_api_calls = False
    elif problem == "missing-pair":
        args.pairs_name = "missing"
    else:
        frozen = json.loads(path.read_text())
        if problem == "stale":
            frozen["fingerprint"] = "stale"
        else:
            frozen["llm_calls"][0]["model"] = "test-ultra"
        path.write_text(json.dumps(frozen))
    tf = NarrativeProvider()
    with pytest.raises((ValueError, PairNotCached)):
        run(args, [case], tf=tf)
    assert not tf.calls
    assert not args.out.exists()


def test_failed_generation_preserves_billed_attempts(experiment):
    args, case, _ = experiment
    with pytest.raises(StructuredOutputError):
        run(args, [case], tf=NarrativeProvider(fail=True))
    assert (args.out / "INCOMPLETE.txt").exists()
    assert not (args.out / "rows.json").exists()
    saved = json.loads((args.out / f"{case.id}--mixed.calls.json").read_text())
    assert saved["new_calls"][0]["cost_usd"] == 0.002
