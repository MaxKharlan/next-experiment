"""Offline comparison of provider usage, job receipts and a read-only ledger snapshot.

Matching aggregate usage is not an invoice or a per-job billing receipt. This
module never settles reservations or changes permission to submit paid work.
"""

import hashlib
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

MILLION = Decimal(1_000_000)
LEDGER_EPSILON = Decimal("0.000000000001")  # Existing SQLite REAL serialization.
TERMINAL = {"succeeded", "failed", "cancelled"}


def money(value):
    if not isinstance(value, str):
        raise TypeError("Amounts must be decimal strings")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("Amounts must be finite and nonnegative")
    return result


def usd(value):
    return format(value.normalize(), "f")


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Timestamps require an explicit timezone")
    return result


def unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"Missing or duplicate {key}")
        result[value] = row
    return result


def identity(value):
    return hashlib.sha256(value.encode()).hexdigest()


def snapshot_ledger(path, bindings):
    """Bindings map operation IDs to known provider job IDs; no recovery writes."""
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        cap = connection.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0]
        rows = connection.execute("SELECT * FROM spending ORDER BY id").fetchall()
    finally:
        connection.close()
    return {
        "cap_usd": str(cap),
        "operations": [
            {
                "operation_id": r["id"],
                "job_key": identity(bindings[r["id"]]) if r["id"] in bindings else None,
                "status": r["status"],
                "reserved_usd": str(r["reserved"]),
                "actual_usd": None if r["actual"] is None else str(r["actual"]),
            }
            for r in rows
        ],
    }


def reconcile(bundle):
    if bundle["schema_version"] != 1 or bundle["currency"] != "USD":
        raise ValueError("Expected schema 1 and USD")
    usage, inventory = bundle["usage"], bundle["inventory"]
    if not usage["scope"] or usage["scope"] != inventory["scope"]:
        raise ValueError("Usage and inventory scopes must agree")
    start, end = timestamp(usage["period_start"]), timestamp(usage["period_end"])
    cutoff = timestamp(usage["updated_at"])
    if not start < cutoff <= end or timestamp(inventory["observed_at"]) < cutoff:
        raise ValueError("Invalid period or snapshot precedes usage cutoff")
    if inventory["pagination_complete"] is not True:
        raise ValueError("Complete provider pagination is required")
    products = unique(usage["products"], "product")
    jobs = unique(inventory["jobs"], "job_key")
    if not jobs or not any(p["kind"] == "training" for p in products.values()):
        raise ValueError("Training reconciliation requires jobs and training products")
    operations = unique(bundle["ledger"]["operations"], "operation_id")
    bound = [o["job_key"] for o in operations.values() if o["job_key"] is not None]
    if len(bound) != len(set(bound)):
        raise ValueError("A provider job cannot settle multiple local operations")
    discrepancies, totals = [], defaultdict(Decimal)
    matched_tokens, matched_jobs = Counter(), Counter()
    job_costs, deferred, unresolved = {}, [], []
    for key, job in jobs.items():
        if job["status"] not in TERMINAL | {"running", "queued", "validating_files"}:
            raise ValueError("Unknown provider status")
        count = job["trained_tokens"]
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("trained_tokens must be a nonnegative integer or null")
        if job["status"] != "succeeded":
            (unresolved if job["status"] in TERMINAL else deferred).append(key)
            continue  # Zero tokens on a running/failed job do not prove zero charge.
        if count is None or job["finished_at"] is None:
            discrepancies.append({"kind": "incomplete_succeeded_job", "job_key": key})
            continue
        finished = timestamp(job["finished_at"])
        if not start <= finished < end or finished > cutoff:
            deferred.append(key)
            continue
        product = job["product"]
        if product not in products or products[product]["kind"] != "training":
            discrepancies.append({"kind": "missing_training_product", "job_key": key})
            continue
        matched_tokens[product] += count
        matched_jobs[product] += 1
        job_costs[key] = (
            Decimal(count) * money(products[product]["rate_usd_per_million"]) / MILLION
        )
    training_rows = []
    for name, row in products.items():
        kind = row["kind"]
        if kind not in {"training", "inference"}:
            raise ValueError("Unsupported usage kind")
        quantity, cost = money(row["quantity_million_tokens"]), money(row["cost_usd"])
        rate = money(row["rate_usd_per_million"])
        if quantity * MILLION != (quantity * MILLION).to_integral_value():
            raise ValueError("Fractional token quantity")
        if quantity * rate != cost:
            discrepancies.append({"kind": "usage_rate_mismatch", "product": name})
        totals[kind] += cost
        if kind == "training":
            expected = int(quantity * MILLION)
            delta = matched_tokens[name] - expected
            if delta:
                discrepancies.append(
                    {
                        "kind": "training_token_mismatch",
                        "product": name,
                        "delta_tokens": delta,
                    }
                )
            training_rows.append(
                {
                    "product": name,
                    "succeeded_jobs": matched_jobs[name],
                    "job_tokens": matched_tokens[name],
                    "usage_tokens": expected,
                    "usage_cost_usd": usd(cost),
                    "delta_tokens": delta,
                }
            )
    local_rows, settled, reserved = [], Decimal(0), Decimal(0)
    for op, row in operations.items():
        hold = money(row["reserved_usd"])
        actual = None if row["actual_usd"] is None else money(row["actual_usd"])
        if row["status"] not in {"settled", "pending"}:
            raise ValueError("Unknown ledger status")
        if (row["status"] == "settled") != (actual is not None):
            raise ValueError("Ledger status and actual cost disagree")
        job_key = row["job_key"]
        estimate = job_costs.get(job_key)
        if actual is None:
            reserved += hold
            outcome = "reservation_retained"
        else:
            settled += actual
            outcome = "pending_usage_evidence"
            if estimate is not None:
                outcome = (
                    "matches_job_token_estimate"
                    if abs(estimate - actual) <= LEDGER_EPSILON
                    else "local_cost_mismatch"
                )
                if outcome == "local_cost_mismatch":
                    discrepancies.append({"kind": outcome, "operation_id": op})
        if job_key not in jobs:
            outcome = "missing_job_binding"
            discrepancies.append({"kind": outcome, "operation_id": op})
        local_rows.append(
            {
                "operation_id": op,
                "outcome": outcome,
                "provider_status": jobs.get(job_key, {}).get("status"),
                "job_token_estimate_usd": None if estimate is None else usd(estimate),
            }
        )
    cap = money(bundle["ledger"]["cap_usd"])
    if settled + reserved > cap + LEDGER_EPSILON:
        discrepancies.append({"kind": "ledger_over_cap"})
    displays = bundle["display_totals"]
    # Rounded UI values are independent observations, not an inferred tax rate.
    usage_total = money(displays["usage_including_tax_usd"])
    transaction_total = money(displays["transactions_including_tax_usd"])
    return {
        "schema_version": 1,
        "currency": "USD",
        "usage_updated_at": usage["updated_at"],
        "aggregate_training_usage_matches": not any(
            d["kind"]
            in {
                "incomplete_succeeded_job",
                "missing_training_product",
                "training_token_mismatch",
                "usage_rate_mismatch",
            }
            for d in discrepancies
        ),
        "invoice_reconciled": False,
        "inference_calls_reconciled": False,
        "inventory_status_counts": dict(
            sorted(Counter(j["status"] for j in jobs.values()).items())
        ),
        "training": training_rows,
        "usage_training_usd": usd(totals["training"]),
        "usage_inference_usd": usd(totals["inference"]),
        "usage_subtotal_usd": usd(totals["training"] + totals["inference"]),
        "display_total_delta_usd": usd(transaction_total - usage_total),
        "deferred_job_count": len(deferred),
        "failed_or_cancelled_job_count": len(unresolved),
        "ledger": {
            "cap_usd": usd(cap),
            "settled_usd": usd(settled),
            "pending_reservations_usd": usd(reserved),
            "committed_usd": usd(settled + reserved),
            "operations": local_rows,
        },
        "discrepancies": discrepancies,
        "limitations": [
            "Usage is an aggregate observation; individual job charges remain token estimates.",
            "Failed/cancelled and deferred jobs are not asserted to have zero cost.",
            "Inference has product totals but no complete per-call provider billing evidence.",
            "Displayed tax and totals are rounded; no tax rate or invoice is inferred.",
            "Report is read-only and cannot release holds or authorize another submission.",
        ],
    }
