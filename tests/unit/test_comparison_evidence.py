from types import SimpleNamespace

import pytest

from core.interventions.evidence import comparison_evidence


@pytest.mark.parametrize(
    ("seeds", "gate", "minimum"),
    [(2, "skipped_unattainable", 1 / 6), (3, "applied", 1 / 20)],
)
def test_gate_resolution_is_explicit(seeds, gate, minimum):
    arm = SimpleNamespace(metrics=[0.5] * seeds)
    evidence = comparison_evidence(arm, arm, min_effect=0.06, noise_sd=None)
    assert evidence["permutation"]["gate"] == gate
    assert evidence["permutation"]["minimum_attainable_p"] == pytest.approx(minimum)
    assert evidence["interval"]["method"] == "pooled_sd_t"
    assert evidence["interval"]["noise_sd"] is None
    assert evidence["resource_matching"] == "not_verified_by_statistical_check"


def test_downloaded_report_preserves_evidence_without_filling_legacy_gaps():
    from core.report.investigation import build_report

    arm = SimpleNamespace(metrics=[0.5, 0.6])
    evidence = comparison_evidence(arm, arm, min_effect=0.06, noise_sd=0.019)
    steps = [
        {"kind": "intervention", "result": {"status": "confirmed"}},
        {"kind": "intervention", "comparison_evidence": evidence},
    ]
    report = build_report(
        {
            "meta": {
                "id": "example",
                "backend": "tf",
                "created_at": "2026-10-19",
                "status": "finished",
            },
            "events": [
                {"type": "step", "data": step, "seq": i} for i, step in enumerate(steps)
            ],
        }
    )
    assert "comparison_evidence" not in report["steps"][0]
    assert report["steps"][1]["comparison_evidence"] == evidence
