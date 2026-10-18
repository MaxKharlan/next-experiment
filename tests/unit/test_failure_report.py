from dataclasses import asdict
from types import SimpleNamespace

import pytest

from bench.failure_report import audit, classify, markdown
from bench.score import Row


def row(**changes):
    values = {
        "case": "fault",
        "method": "agent",
        "truth": ["lr_too_low"],
        "status": "confirmed",
        "cause": "lr_too_low",
        "top": ["lr_too_low"],
        "top1": True,
        "top3": True,
        "p_true": 0.8,
        "false_alarm": None,
        "cost_usd": 0.1,
        "llm_tokens": 100,
        "mini_jobs": 4,
    }
    values.update(changes)
    return Row(**values)


def test_unknown_fix_is_not_a_failure():
    assert classify(row(fix_success=False)) == []
    assert classify(row(fix_success=None)) == []
    assert classify(row(fix_success=False, fixed_metric=0.2, reference=0.5)) == [
        "measured_fix_failed"
    ]


def test_inconclusive_is_distinct_from_measured_fix_failure():
    assert classify(row(status="inconclusive", cause=None, fix_success=False)) == [
        "cause_not_confirmed"
    ]


def test_secondary_wrong_confirmation_is_retained():
    assert classify(row(wrong_confirmations=1)) == ["wrong_confirmations"]


def test_counts_are_per_method_and_findings_keep_evidence():
    cases = [SimpleNamespace(id="fault", truth=["lr_too_low"])]
    report = audit([asdict(row(wrong_confirmations=2))], cases)
    assert report["methods"]["agent"]["wrong_confirmation_count"] == 2
    assert (
        report["methods"]["agent"]["category_case_counts"]["wrong_confirmations"] == 1
    )
    assert report["findings"][0]["row"]["cause"] == "lr_too_low"
    assert report["findings"][0]["source_record"] == 1
    assert "agent" in markdown(report)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "truth", "flags", "nan"])
def test_inconsistent_inputs_rejected(problem):
    cases = [SimpleNamespace(id="fault", truth=["lr_too_low"])]
    records = [asdict(row())]
    if problem == "duplicate":
        records *= 2
    elif problem == "missing":
        cases.append(SimpleNamespace(id="other", truth=[]))
    elif problem == "truth":
        records[0]["truth"] = []
    elif problem == "flags":
        records[0]["top1"] = False
    else:
        records[0]["cost_usd"] = float("nan")
    with pytest.raises(ValueError):
        audit(records, cases)


def test_cli_writes_evidence_and_refuses_overwrite(tmp_path, monkeypatch):
    import json
    import sys

    from bench import failure_report

    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps(asdict(row(wrong_confirmations=1))) + "\n")
    out = tmp_path / "report"
    monkeypatch.setattr(
        failure_report,
        "load_cases",
        lambda split: [SimpleNamespace(id="fault", truth=["lr_too_low"])],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["failure_report", str(source), "--split", "dev", "--out", str(out)],
    )
    failure_report.main()
    report = json.loads((out / "report.json").read_text())
    assert report["split"] == "dev"
    assert len(report["source_sha256"]) == 64
    assert (out / "README.md").exists() and (out / "rows.json").exists()
    original = (out / "report.json").read_bytes()
    with pytest.raises(SystemExit):
        failure_report.main()
    assert (out / "report.json").read_bytes() == original
