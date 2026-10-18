import json

import pytest

from bench import methods
from core.agent.context import InvestigationContext
from core.probes import weight_cache
from tests.unit.test_bench import _case


@pytest.fixture
def cached_context(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXT_EXPERIMENT_WEIGHTS", "1")
    monkeypatch.setattr(weight_cache, "CACHE_DIR", tmp_path)
    ctx = InvestigationContext(
        "test",
        "A",
        "ftjob-test",
        {
            "model": weight_cache.SUPPORTED_MODEL,
            "hyperparameters": {"learning_rate": 1e-5},
        },
    )
    record = {
        "version": weight_cache.CACHE_VERSION,
        "job_id": ctx.job_id,
        "model": weight_cache.SUPPORTED_MODEL,
        "hyperparameters": ctx.job_config["hyperparameters"],
        "adapter_sha256": "a" * 64,
        "raw": {
            "model": weight_cache.SUPPORTED_MODEL,
            "rel_median": 0.005,
            "modules": 196,
        },
    }
    weight_cache.cache_path(ctx.job_id).write_text(json.dumps(record))
    return ctx


def test_cached_probe_reads_the_correct_job(cached_context):
    probe = weight_cache.investigation_probes(cached_context)[-1]
    assert probe.name == "lora_weights"
    result = probe.run(cached_context)
    assert result.outcome == "small"
    assert json.loads(result.summary)["source_job_id"] == cached_context.job_id


def test_disabled_probe_needs_no_cache(monkeypatch):
    monkeypatch.setenv("NEXT_EXPERIMENT_WEIGHTS", "0")
    ctx = InvestigationContext("test", "A", "missing", {})
    assert all(p.name != "lora_weights" for p in weight_cache.investigation_probes(ctx))


def test_enabled_probe_requires_a_matching_cache(cached_context):
    cached_context.job_id = "missing"
    with pytest.raises(FileNotFoundError, match="No B25 measurement"):
        weight_cache.investigation_probes(cached_context)


def test_cache_rejects_different_recipe(cached_context):
    cached_context.job_config["hyperparameters"] = {"learning_rate": 0.001}
    with pytest.raises(ValueError, match="does not match"):
        weight_cache.investigation_probes(cached_context)


@pytest.mark.parametrize("method", ["react", "checklist", "next_experiment"])
def test_benchmark_methods_receive_the_same_probes(tmp_path, monkeypatch, method):
    case = _case(tmp_path)
    expected = (object(),)
    received = []

    def probes(ctx):
        assert ctx.job_id == case.job_id
        assert ctx.job_config["model"] == case.model
        return expected

    def diagnose(*args, **kwargs):
        received.append(kwargs["probes"])
        return "result"

    monkeypatch.setattr(methods, "investigation_probes", probes)
    monkeypatch.setattr(methods, "diagnose_react", diagnose)
    monkeypatch.setattr(methods, "diagnose_checklist", diagnose)
    monkeypatch.setattr(methods, "investigate", diagnose)
    monkeypatch.setattr(methods, "from_investigation", lambda result: result)

    result = methods.run_method(method, case, None, None, pair_cost_usd=0.1)
    assert result == "result"
    assert received == [expected]
