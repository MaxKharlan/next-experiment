"""Verify and export the completed format/content development evidence."""

from __future__ import annotations

import argparse
import json
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


def export(root, output):
    plan, state, digest = load(root)
    decision = json.loads((root / "decision.json").read_text())
    if decision["protocol_sha256"] != digest:
        raise ValueError("Decision binding mismatch")
    summaries = {}
    for name, entry in {"base": {}, **state}.items():
        path = root / "scores" / f"{name}.json"
        record = checked_score(path, plan, digest, entry)
        summaries[name] = {}
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
            summaries[name][kind] = {
                **record[kind]["summary"],
                "n": len(rows),
                "hit_limit": sum(r["hit_limit"] for r in rows),
                "families": families,
                "descriptive_unwrapped_field_accuracy_all_examples": {
                    k: v / len(rows) for k, v in fields.items()
                },
            }
    ledger = SpendingLedger(root / "spending.sqlite3", 5).snapshot()
    if (
        len(state) != 12
        or len(summaries) != 13
        or any(r["status"] != "settled" for r in ledger["operations"])
    ):
        raise ValueError("Incomplete models or spending reconciliation")
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
    state = json.loads((output / "state.json").read_text())
    decision = json.loads((output / "decision.json").read_text())
    if set(state) != set(plan["jobs"]) or len(state) != 12:
        raise ValueError("Incomplete exported cohort")
    groups = {a: {k: {} for k in ("format", "content")} for a in decision["groups"]}
    for name, entry in {"base": {}, **state}.items():
        record = checked_score(output / "scores" / f"{name}.json", plan, digest, entry)
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.verify:
        verify_export(args.out)
    else:
        export(args.root, args.out)


if __name__ == "__main__":
    main()
