import pytest

from core.db.spending import SpendingLedger
from core.eval.data import Example
from core.eval.metrics import exact_match
from scripts.execute_reference_grid import settle_estimate
from scripts.score_reference_grid import choose, verify_rows


def test_cost_reconciliation_after_crash_is_idempotent(tmp_path):
    ledger = SpendingLedger(tmp_path / "budget.sqlite3", 2)
    ledger.reserve("one", "training", 1)
    settle_estimate(ledger, "one", 0.5)
    settle_estimate(ledger, "one", 0.5)
    assert ledger.snapshot()["committed_usd"] == 0.5
    with pytest.raises(ValueError, match="Conflicting"):
        settle_estimate(ledger, "one", 0.4)


def test_selection_needs_full_grid_and_breaks_ties_by_frozen_index():
    jobs = {"g0": {"grid_index": 0}, "g1": {"grid_index": 1}}
    with pytest.raises(ValueError):
        choose({"g0": 1}, jobs)
    assert choose({"g1": 0.9, "g0": 0.9}, jobs) == "g0"
    assert choose({"g1": 1, "g0": 0.9}, jobs) == "g1"


def test_resume_rejects_wrong_identity_and_incomplete_complete_flag():
    examples = [Example(1, [], "a", None)]
    record = {"protocol": "one", "rows": [], "complete": True}
    with pytest.raises(ValueError, match="Incomplete"):
        verify_rows(record, examples, {"protocol": "one"}, exact_match)
    record.update(
        complete=False,
        rows=[
            {
                "line": 2,
                "expected": "a",
                "source": None,
                "score": 1,
                "answer": "a",
                "hit_limit": False,
                "stopped_on_eos": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="identity"):
        verify_rows(record, examples, {"protocol": "one"}, exact_match)
