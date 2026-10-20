import json
from copy import deepcopy
from pathlib import Path

import pytest

from core.db.spending import SpendingLedger
from core.report.cost_reconciliation import identity, reconcile, snapshot_ledger


@pytest.fixture
def bundle():
    return {
        "schema_version": 1,
        "currency": "USD",
        "usage": {
            "scope": "test-project",
            "period_start": "2026-10-01T00:00:00Z",
            "period_end": "2026-11-01T00:00:00Z",
            "updated_at": "2026-10-19T12:22:00Z",
            "products": [
                {
                    "product": "training",
                    "kind": "training",
                    "quantity_million_tokens": "0.25",
                    "cost_usd": "0.10",
                    "rate_usd_per_million": "0.4",
                },
                {
                    "product": "input",
                    "kind": "inference",
                    "quantity_million_tokens": "0.01",
                    "cost_usd": "0.003",
                    "rate_usd_per_million": "0.3",
                },
            ],
        },
        "inventory": {
            "scope": "test-project",
            "pagination_complete": True,
            "observed_at": "2026-10-19T13:00:00Z",
            "jobs": [
                {
                    "job_key": "job-1",
                    "product": "training",
                    "status": "succeeded",
                    "trained_tokens": 250000,
                    "finished_at": "2026-10-18T12:00:00Z",
                },
                {
                    "job_key": "job-2",
                    "product": "training",
                    "status": "running",
                    "trained_tokens": 0,
                    "finished_at": None,
                },
            ],
        },
        "ledger": {
            "cap_usd": "5",
            "operations": [
                {
                    "operation_id": "one",
                    "job_key": "job-1",
                    "status": "settled",
                    "reserved_usd": "0.12",
                    "actual_usd": "0.10000000000000001",
                },
                {
                    "operation_id": "two",
                    "job_key": "job-2",
                    "status": "pending",
                    "reserved_usd": "0.12",
                    "actual_usd": None,
                },
            ],
        },
        "display_totals": {
            "usage_including_tax_usd": "0.11",
            "transactions_including_tax_usd": "0.12",
        },
    }


def test_exact_usage_with_pending_hold_and_distinct_invoice_status(bundle):
    before = deepcopy(bundle)
    report = reconcile(bundle)
    assert bundle == before
    assert report["aggregate_training_usage_matches"]
    assert not report["invoice_reconciled"]
    assert not report["inference_calls_reconciled"]
    assert report["usage_subtotal_usd"] == "0.103"
    assert report["display_total_delta_usd"] == "0.01"
    assert report["ledger"]["pending_reservations_usd"] == "0.12"
    assert report["ledger"]["operations"][0]["outcome"] == "matches_job_token_estimate"
    assert report["ledger"]["operations"][1]["job_token_estimate_usd"] is None
    assert report["discrepancies"] == []


@pytest.mark.parametrize(
    "collection,key",
    [("inventory", "jobs"), ("usage", "products"), ("ledger", "operations")],
)
def test_duplicate_evidence_rejected(bundle, collection, key):
    bundle[collection][key].append(deepcopy(bundle[collection][key][0]))
    with pytest.raises(ValueError, match="duplicate"):
        reconcile(bundle)


def test_cannot_bill_same_job_twice(bundle):
    bundle["ledger"]["operations"][1]["job_key"] = "job-1"
    with pytest.raises(ValueError, match="multiple local"):
        reconcile(bundle)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("inventory", "pagination_complete", False),
        ("inventory", "scope", "another-project"),
        ("inventory", "observed_at", "2026-10-19T10:00:00Z"),
        ("usage", "updated_at", "2026-12-01T00:00:00Z"),
        ("usage", "updated_at", "2026-10-19T12:22:00"),
    ],
)
def test_scope_cutoff_and_incomplete_pagination_fail(bundle, section, key, value):
    bundle[section][key] = value
    with pytest.raises(ValueError):
        reconcile(bundle)


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-1", "not-money", 0.1, True])
def test_invalid_financial_values(bundle, amount):
    bundle["usage"]["products"][0]["cost_usd"] = amount
    with pytest.raises((ValueError, TypeError)):
        reconcile(bundle)


def test_currency_must_be_explicit(bundle):
    bundle["currency"] = "EUR"
    with pytest.raises(ValueError, match="USD"):
        reconcile(bundle)


def test_one_token_mismatch_is_visible(bundle):
    bundle["inventory"]["jobs"][0]["trained_tokens"] -= 1
    report = reconcile(bundle)
    assert not report["aggregate_training_usage_matches"]
    assert report["training"][0]["delta_tokens"] == -1
    assert {d["kind"] for d in report["discrepancies"]} == {
        "training_token_mismatch",
        "local_cost_mismatch",
    }


def test_missing_binding_does_not_release_hold(bundle):
    bundle["ledger"]["operations"][1]["job_key"] = None
    report = reconcile(bundle)
    assert report["ledger"]["pending_reservations_usd"] == "0.12"
    assert report["discrepancies"] == [
        {"kind": "missing_job_binding", "operation_id": "two"}
    ]


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_terminal_failure_zero_tokens_not_claimed_free(bundle, status):
    bundle["inventory"]["jobs"][1]["status"] = status
    report = reconcile(bundle)
    assert report["failed_or_cancelled_job_count"] == 1
    assert report["ledger"]["pending_reservations_usd"] == "0.12"
    assert report["ledger"]["operations"][1]["job_token_estimate_usd"] is None


@pytest.mark.parametrize(
    "finished", ["2026-10-19T12:23:00Z", "2026-09-30T23:59:59Z", "2026-11-01T00:00:00Z"]
)
def test_jobs_outside_usage_window_deferred(bundle, finished):
    bundle["inventory"]["jobs"][0]["finished_at"] = finished
    report = reconcile(bundle)
    assert not report["aggregate_training_usage_matches"]
    assert report["deferred_job_count"] == 2
    assert report["ledger"]["operations"][0]["job_token_estimate_usd"] is None


def test_missing_tokens_are_unknown_not_zero(bundle):
    bundle["inventory"]["jobs"][0]["trained_tokens"] = None
    report = reconcile(bundle)
    assert not report["aggregate_training_usage_matches"]
    assert report["discrepancies"][0]["kind"] == "incomplete_succeeded_job"


def test_usage_amount_mismatch(bundle):
    bundle["usage"]["products"][0]["cost_usd"] = "0.11"
    report = reconcile(bundle)
    assert not report["aggregate_training_usage_matches"]
    assert report["discrepancies"][0]["kind"] == "usage_rate_mismatch"


def test_snapshot_does_not_mutate_sqlite_or_create_missing_database(tmp_path):
    path = tmp_path / "spending.sqlite3"
    ledger = SpendingLedger(path, 5)
    ledger.reserve("operation", "training", 0.12)
    before = path.read_bytes()
    snapshot = snapshot_ledger(path, {"operation": "provider-id"})
    assert path.read_bytes() == before
    assert snapshot["operations"][0]["job_key"] == identity("provider-id")
    assert snapshot["operations"][0]["actual_usd"] is None
    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        snapshot_ledger(tmp_path / "absent.sqlite3", {})
    assert not (tmp_path / "absent.sqlite3").exists()


def test_published_snapshot_reproduces_report():
    root = (
        Path(__file__).resolve().parents[1]
        / "docs/results/provider-cost-reconciliation-20261019"
    )
    report = reconcile(json.loads((root / "input.json").read_text()))
    saved = json.loads((root / "report.json").read_text())
    saved.pop("input_sha256")
    assert report == saved
    assert sum(r["succeeded_jobs"] for r in report["training"]) == 224
    assert report["usage_training_usd"] == "8.0379272"
    assert (
        sum(
            r["outcome"] == "matches_job_token_estimate"
            for r in report["ledger"]["operations"]
        )
        == 6
    )
