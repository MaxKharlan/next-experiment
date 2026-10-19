"""Verify and export the completed format/content development evidence."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from bench.tasks.structured import sha
from core.db.spending import SpendingLedger
from core.interventions.stats import noise_floor
from scripts.format_content_development import checked_score, load
from scripts.fresh_answer_calibration import save


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
            summaries[name][kind] = {
                **record[kind]["summary"],
                "n": len(rows),
                "hit_limit": sum(r["hit_limit"] for r in rows),
                "families": families,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.out)


if __name__ == "__main__":
    main()
