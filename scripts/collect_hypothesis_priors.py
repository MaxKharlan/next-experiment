"""Collect one fresh Super prior draw per benchmark case with a hard local ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bench.cases import load_cases
from core.agent.router import Router
from core.agent.schemas import HypothesisList
from core.agent.steps import propose_hypotheses
from core.config import Settings
from core.connectors.token_factory import TokenFactory
from core.db.spending import SpendingLedger
from core.telemetry.calls import CallLog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap-usd", type=float, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory already exists")
    args.out.mkdir(parents=True)
    ledger = SpendingLedger(args.out / "spending.sqlite3", args.cap_usd)
    settings = Settings(_env_file=args.env_file) if args.env_file else Settings()
    tf = TokenFactory.from_settings(settings)
    tf.spending = ledger
    log = CallLog(f"prior-{args.split}")
    router = Router(tf, log)
    hashes = {}
    for case in load_cases(args.split):
        hypotheses = propose_hypotheses(router, case.case, mode="A")
        payload = HypothesisList(hypotheses=hypotheses).model_dump_json(indent=2) + "\n"
        path = args.out / f"{case.id}--draw-1.json"
        path.write_text(payload, encoding="utf-8")
        hashes[case.id] = hashlib.sha256(payload.encode()).hexdigest()
        print(f"{case.id}: {len(hypotheses)} hypotheses", flush=True)
    summary = {
        "split": args.split,
        "case_count": len(hashes),
        "draw_sha256": hashes,
        "llm": log.totals("model"),
        "spending": ledger.snapshot(),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["spending"], indent=2))


if __name__ == "__main__":
    main()
