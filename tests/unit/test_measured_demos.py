import hashlib
import json
from pathlib import Path

import pytest

from apps.api import runners
from scripts.build_measured_demos import build_cases

ROOT = Path(__file__).resolve().parents[2]


def test_saved_experiments_reproduce_measured_effects_and_provenance():
    cases = build_cases()
    expected = {
        "answer-cut-kaggle": (12, 0.18311688311688323, True),
        "answer-cut-tf": (6, 0.19740259740259736, False),
    }
    for name, (count, effect, gate) in expected.items():
        saved = cases[name]
        assert saved == json.loads((ROOT / "demo" / f"{name}.json").read_text())
        report = saved["report"]
        m = report["measured_experiment"]
        assert report["status"] == "measured"
        assert not report.get("verdict") and not report["hypotheses"]
        assert len(m["runs"]) == count
        assert m["causal_result"]["effect"] == pytest.approx(effect)
        assert m["means"]["target"] < m["means"]["healthy"]
        assert m["permutation_gate_applied"] is gate
        for source in m["sources"]:
            assert (
                source["sha256"]
                == hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
            )
        summary = runners.replay_events(name)[-1][2]
        assert summary["measured_experiment"] == m
        assert (
            summary["estimated_training_cost_usd"] == m["estimated_training_cost_usd"]
        )
    tf = cases["answer-cut-tf"]["report"]["measured_experiment"]
    assert len({r["job_id"] for r in tf["runs"]}) == 6
    assert len({r["adapter_sha256"] for r in tf["runs"]}) == 6
    assert tf["estimated_training_cost_usd"] == pytest.approx(0.9269888)


def test_measured_experiments_have_their_own_catalog_category():
    index = {c["name"]: c for c in runners.demo_index()}
    for name in ("answer-cut-kaggle", "answer-cut-tf"):
        assert index[name]["category"] == "Measured experiments"
        assert index[name]["status"] == "measured"
    assert index["fault6-tf"]["category"] == "Agent investigations"
    assert index["json_extraction-fault1-audit"]["category"] == "Local audits"


def test_inconsistent_provider_job_is_rejected(monkeypatch):
    from scripts import build_measured_demos as builder

    read = builder._read

    def corrupt(root, path):
        result = read(root, path)
        if path.endswith("/jobs.json"):
            result["jobs"]["healthy-s0"]["id"] = "wrong-job"
        return result

    monkeypatch.setattr(builder, "_read", corrupt)
    with pytest.raises(ValueError, match="provenance mismatch"):
        builder.build_cases()
