"""Verify and export the completed format/content development evidence."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

from bench.tasks.structured import sha
from core.db.spending import SpendingLedger
from core.eval.format_content import unwrap
from core.eval.structured import validate_answer
from core.interventions.stats import noise_floor
from scripts.format_content_development import checked_score, gate_failures, load
from scripts.fresh_answer_calibration import save
from scripts.fresh_answer_power import decide as power_decide


def accounting(
    plan, state, ledger, inherited, provenance, digest, allow_cancelled_holds=False
):
    """Validate paid-job lineage without treating cancelled holds as zero bills."""

    def money(value, expected):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("Accounting amount mismatch")

    if set(state) != set(plan["jobs"]) or len(state) != 12:
        raise ValueError("Incomplete models")
    operations = {r["id"]: r for r in ledger["operations"]}
    if len(operations) != len(ledger["operations"]):
        raise ValueError("Duplicate spending operation")
    inherited_ids = {r["id"] for r in inherited["operations"]}
    for row in inherited["operations"]:
        if row["status"] != "settled" or operations.get(row["id"]) != row:
            raise ValueError("Inherited accounting changed")
    retries = any(e.get("previous_attempts") for e in state.values())
    if retries and (
        not provenance
        or provenance.get("protocol_sha256") != digest
        or set(provenance.get("jobs", {})) != set(state)
        or provenance.get("experimental_slots") != 12
        or provenance.get("provider_submissions") != 24
        or provenance.get("authorized_infrastructure_retries") != 1
        or provenance.get("invoice_reconciled") is not False
    ):
        raise ValueError("Missing or invalid retry provenance")
    expected_ids = set(inherited_ids)
    job_ids = set()
    pending_ids = set()
    training_cost = 0.0
    for name, entry in state.items():
        spec = plan["jobs"][name]
        if (
            entry.get("status") != "succeeded"
            or entry.get("settled") is not True
            or entry.get("protocol_sha256") != digest
            or not entry.get("weights_sha256")
            or type(entry.get("trained_tokens")) is not int
            or entry["trained_tokens"] < 0
        ):
            raise ValueError("Incomplete successful model")
        operation = operations[entry["operation_id"]]
        cost = entry["trained_tokens"] * plan["price"]["usd_per_million"] / 1e6
        if operation["status"] != "settled":
            raise ValueError("Successful model spending is pending")
        money(operation["actual_usd"], cost)
        money(entry["estimated_cost_usd"], cost)
        money(operation["reserved_usd"], spec["reservation_usd"])
        training_cost += cost
        attempts = [("replacement", entry)] if retries else [("original", entry)]
        if retries:
            if len(entry.get("previous_attempts", [])) != 1:
                raise ValueError("Expected exactly one infrastructure retry")
            old = entry["previous_attempts"][0]
            if old.get("status") != "cancelled" or old.get("protocol_sha256") != digest:
                raise ValueError("Original attempt is not cancelled")
            attempts.insert(0, ("original", old))
            old_operation = operations[old["operation_id"]]
            # This mode publishes retained holds; it cannot certify their release.
            if (
                not allow_cancelled_holds
                or old_operation["status"] != "pending"
                or old_operation["actual_usd"] is not None
                or provenance.get("old_reservations_retained") is not True
            ):
                raise ValueError(
                    "Cancelled billing remains unresolved; explicit retained-hold export required"
                )
            money(old_operation["reserved_usd"], spec["reservation_usd"])
            pending_ids.add(old["operation_id"])
        for label, attempt in attempts:
            if attempt["operation_id"] in expected_ids or attempt["job_id"] in job_ids:
                raise ValueError("Duplicate job or operation lineage")
            expected_ids.add(attempt["operation_id"])
            job_ids.add(attempt["job_id"])
            if not retries:
                continue
            receipt = provenance["jobs"][name][label]
            if (
                receipt["id"] != attempt["job_id"]
                or receipt["operation_id"] != attempt["operation_id"]
                or receipt["status"] != attempt["status"]
                or receipt["model"] != plan["model"]
                or receipt["seed"] != spec["seed"]
                or receipt["hyperparameters"] != plan["recipe"]
                or not receipt.get("finished_at")
                or any(
                    receipt[k] != attempt[k] or attempt[k] != entry[k]
                    for k in ("training_file", "validation_file")
                )
            ):
                raise ValueError("Provider provenance mismatch")
            if label == "original":
                if receipt["trained_tokens"] != 0 or receipt["trained_steps"] != 0:
                    raise ValueError("Cancelled original has training progress")
            elif receipt["trained_tokens"] != entry["trained_tokens"]:
                raise ValueError("Provider token count mismatch")
    if set(operations) != expected_ids:
        raise ValueError("Unmatched spending operations")
    if {
        r["id"] for r in operations.values() if r["status"] == "pending"
    } != pending_ids:
        raise ValueError("Unexpected pending spending")
    settled = sum(
        r["actual_usd"] for r in operations.values() if r["status"] == "settled"
    )
    held = sum(operations[i]["reserved_usd"] for i in pending_ids)
    money(ledger["cap_usd"], plan["cap_usd"])
    money(ledger["committed_usd"], settled + held)
    money(ledger["available_usd"], max(0, plan["cap_usd"] - settled - held))
    if settled + held > plan["cap_usd"]:
        raise ValueError("Budget exceeded")
    return {
        "status": "cancelled_holds_pending"
        if pending_ids
        else "token_estimates_settled",
        "invoice_reconciled": False,
        "experimental_slots": 12,
        "provider_submissions": len(job_ids),
        "cancelled_originals": len(pending_ids),
        "pending_operation_ids": sorted(pending_ids),
        "retained_holds_usd": held,
        "new_training_token_estimate_usd": training_cost,
        "settled_token_estimate_including_inherited_usd": settled,
        "committed_including_holds_usd": settled + held,
    }


def model_summary(record):
    result = {}
    for kind in ("format", "content"):
        rows = record[kind]["rows"]
        families = {}
        for source in sorted({r["source"] for r in rows}):
            subset = [r for r in rows if r["source"] == source]
            families[source] = {
                "n": len(subset),
                **{
                    key: sum(r[key] for r in subset) / len(subset)
                    for key in record[kind]["summary"]
                },
            }
        fields = {
            key: 0
            for key in ("callback", "due_date", "priority", "product", "ticket_id")
        }
        for row in rows:
            try:
                value = validate_answer("json_extraction", unwrap(row["answer"]))
            except (ValueError, TypeError, RecursionError):
                continue
            expected = json.loads(row["expected"])
            for key in fields:
                fields[key] += value[key] == expected[key]
        result[kind] = {
            **record[kind]["summary"],
            "n": len(rows),
            "hit_limit": sum(r["hit_limit"] for r in rows),
            "families": families,
            "descriptive_unwrapped_field_accuracy_all_examples": {
                k: v / len(rows) for k, v in fields.items()
            },
        }
    return result


def same_accounting(saved, computed):
    """Ignore only float summation roundoff, never lineage or retained holds."""
    if saved.keys() != computed.keys():
        return False
    for key, expected in computed.items():
        value = saved[key]
        if key.endswith("_usd"):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)
            ):
                return False
        elif value != expected:
            return False
    return True


def export(root, output, allow_cancelled_holds=False):
    plan, state, digest = load(root)
    decision = json.loads((root / "decision.json").read_text())
    if decision["protocol_sha256"] != digest:
        raise ValueError("Decision binding mismatch")
    summaries = {}
    for name, entry in {"base": {}, **state}.items():
        path = root / "scores" / f"{name}.json"
        record = checked_score(path, plan, digest, entry)
        summaries[name] = model_summary(record)
    ledger = SpendingLedger(root / "spending.sqlite3", 5).snapshot()
    provenance_path = root / "restart-20261019-r1/terminal-provenance.json"
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    )
    account = accounting(
        plan,
        state,
        ledger,
        json.loads((root / "inherited-spending.json").read_text()),
        provenance,
        digest,
        allow_cancelled_holds,
    )
    if len(summaries) != 13:
        raise ValueError("Incomplete model scores")
    output.mkdir(parents=True, exist_ok=False)
    files = [
        root / n
        for n in (
            "protocol.json",
            "protocol.sha256",
            "state.json",
            "inherited-spending.json",
            "completion.json",
            "decision.json",
        )
    ]
    files += sorted((root / "scores").glob("*.json"))
    files += sorted((root / "data").glob("*"))
    files += sorted(root.glob("*-s*-train.jsonl"))
    for path in files:
        dest = output / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    save(output / "spending.json", ledger)
    save(output / "accounting.json", account)
    if provenance is not None:
        save(output / "restart-provenance.json", provenance)
    noise = {
        metric: asdict(
            noise_floor(
                {
                    arm: decision["groups"][arm]["content"][metric]
                    for arm in ("healthy", "cut")
                }
            )
        )
        for metric in ("strict_exact", "choice_exact")
    }
    save(
        output / "summary.json",
        {
            "protocol_sha256": digest,
            "models": summaries,
            "healthy_cut_descriptive_noise": noise,
            "decision": decision,
            "spending": ledger,
            "accounting": account,
            "boundary": "All results are development evidence. Descriptive noise is reported even if admission fails; it does not override gates or select confirmation seeds. Costs are provider-token estimates, not invoices.",
        },
    )
    save(
        output / "manifest.json",
        {
            str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    print(
        json.dumps(
            {
                "models": len(summaries),
                "files": len(files),
                "committed_usd": ledger["committed_usd"],
                "output": str(output),
            },
            indent=2,
        )
    )


def verify_export(output):
    """Rescore the public bundle without model weights, credentials or local paths."""
    manifest = json.loads((output / "manifest.json").read_text())
    actual_files = {
        str(p.relative_to(output))
        for p in output.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    if set(manifest) != actual_files:
        raise ValueError("Export manifest coverage mismatch")
    for name, expected in manifest.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or sha(path) != expected:
            raise ValueError("Export manifest mismatch")
    digest = (output / "protocol.sha256").read_text().strip()
    if sha(output / "protocol.json") != digest:
        raise ValueError("Protocol digest mismatch")
    plan = json.loads((output / "protocol.json").read_text())
    for entry in plan["datasets"].values():
        path = output / "data" / Path(entry["path"]).name
        if sha(path) != entry["sha256"]:
            raise ValueError("Exported dataset changed")
        entry["path"] = str(path)
    for job in plan["jobs"].values():
        for role in ("train", "validation"):
            path = (
                output / ("data" if role == "validation" else "") / Path(job[role]).name
            )
            if sha(path) != job[f"{role}_sha256"]:
                raise ValueError("Exported training data changed")
    for name, expected in plan["protected_sha256"].items():
        original = Path(name)
        path = (
            output / ("data" if original.parent.name == "data" else "") / original.name
        )
        if sha(path) != expected:
            raise ValueError("Protected export evidence changed")
    state = json.loads((output / "state.json").read_text())
    decision = json.loads((output / "decision.json").read_text())
    if set(state) != set(plan["jobs"]) or len(state) != 12:
        raise ValueError("Incomplete exported cohort")
    if decision["protocol_sha256"] != digest:
        raise ValueError("Decision binding mismatch")
    completion = json.loads((output / "completion.json").read_text())
    if completion != {
        "protocol_sha256": digest,
        "status": decision["status"],
        "paid_jobs": 12,
        "scored_models": 13,
    }:
        raise ValueError("Completion mismatch")
    provenance_path = output / "restart-provenance.json"
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    )
    ledger = json.loads((output / "spending.json").read_text())
    account = accounting(
        plan,
        state,
        ledger,
        json.loads((output / "inherited-spending.json").read_text()),
        provenance,
        digest,
        allow_cancelled_holds=True,
    )
    if not same_accounting(
        json.loads((output / "accounting.json").read_text()), account
    ):
        raise ValueError("Accounting summary mismatch")
    report = json.loads((output / "summary.json").read_text())
    if (
        report["protocol_sha256"] != digest
        or report["decision"] != decision
        or not same_accounting(report["accounting"], account)
        or report["spending"] != ledger
        or set(report["models"]) != {"base", *state}
    ):
        raise ValueError("Published summary mismatch")
    groups = {
        a: {k: {} for k in ("format", "content")}
        for a in ("cut", "format", "healthy", "wrong")
    }
    for name, entry in {"base": {}, **state}.items():
        record = checked_score(output / "scores" / f"{name}.json", plan, digest, entry)
        if report["models"][name] != model_summary(record):
            raise ValueError("Published model summary mismatch")
        if name == "base":
            if decision["base"] != {
                k: record[k]["summary"] for k in ("format", "content")
            }:
                raise ValueError("Base decision mismatch")
            continue
        arm = name.split("-", 1)[0]
        for kind in ("format", "content"):
            for metric, value in record[kind]["summary"].items():
                groups[arm][kind].setdefault(metric, []).append(value)
    noise = {
        metric: asdict(
            noise_floor(
                {arm: groups[arm]["content"][metric] for arm in ("healthy", "cut")}
            )
        )
        for metric in ("strict_exact", "choice_exact")
    }
    if report["healthy_cut_descriptive_noise"] != noise:
        raise ValueError("Published noise summary mismatch")
    failures = gate_failures(groups, plan["gates"])
    digests = {
        a: [
            entry["weights_sha256"]
            for name, entry in state.items()
            if name.startswith(a + "-")
        ]
        for a in groups
    }
    if any(len(set(values)) != 3 for values in digests.values()):
        failures.append("identical_weights")
    if (
        groups != decision["groups"]
        or failures != decision["failures"]
        or digests != decision["weight_digests"]
    ):
        raise ValueError("Decision cannot be reproduced")
    if failures and (
        decision["status"] != "stopped"
        or decision["selected_seeds"]
        or decision["power"] is not None
    ):
        raise ValueError("Failed gate did not stop selection")
    if not failures:
        power = power_decide(
            {
                "healthy": groups["healthy"]["content"]["strict_exact"],
                "damaged": groups["cut"]["content"]["strict_exact"],
            },
            {"healthy": digests["healthy"], "damaged": digests["cut"]},
            plan["power_rule"],
        )
        if (
            power != decision["power"]
            or power["status"] != decision["status"]
            or power["selected_seeds"] != decision["selected_seeds"]
        ):
            raise ValueError("Power decision differs")
    print(
        json.dumps(
            {
                "verified_models": 13,
                "verified_predictions": 13 * 72,
                "verified_choice_decisions": 13 * 48,
                "provider_submissions": account["provider_submissions"],
                "accounting_status": account["status"],
                "gate_failures": failures,
                "protocol_sha256": digest,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--allow-cancelled-holds",
        action="store_true",
        help="Publish completed science with proven cancelled originals' holds retained",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.verify:
        verify_export(args.out)
    else:
        export(args.root, args.out, args.allow_cancelled_holds)


if __name__ == "__main__":
    main()
