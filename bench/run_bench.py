"""The benchmark: every method on every case of a split, scored as in section 10.

    uv run --group ml --group eval python -m bench.run_bench --split dev --no-train

For each case, pairs and fixes are cached under artifacts/bench/<split>/<case id>/,
so a rerun and the other methods reuse them; the investigations listed in the
case's "reports" seed both caches. A pair or a fix that is not cached trains on
Token Factory, unless --no-train: then a method that needs a missing pair skips the
case, and a missing fix leaves fix success unknown. Diagnoses go next to the caches,
the rows to artifacts/bench/<split>/rows.jsonl and the table to summary.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from bench.cases import BenchCase, load_cases
from bench.diagnosis import Diagnosis
from bench.fixes import (
    FixCache,
    fix_key,
    fixed_hyperparameters,
    reference_metric,
    seed_fixes,
)
from bench.methods import METHODS, pair_cost, run_method, tf_pair_runner
from bench.pair_cache import PairCache, PairNotCached, seed_cache
from bench.reference import ReferenceRun
from bench.score import Row, score, summarize, table
from core.agent.loop import RunPair
from core.agent.router import Router
from core.config import get_settings
from core.connectors.storage import LocalStorage
from core.connectors.tavily import Tavily
from core.connectors.token_factory import TokenFactory
from core.telemetry.calls import CallLog


def run_case(
    case: BenchCase,
    methods: Sequence[str],
    *,
    router_for: Callable[[str], Router],
    pairs: RunPair,
    fixes: FixCache,
    reference: float | None,
    pair_cost_usd: float,
    tavily: Tavily | None = None,
    on_diagnosis: Callable[[Diagnosis], None] | None = None,
) -> list[Row]:
    rows = []
    for method in methods:
        try:
            d = run_method(
                method,
                case,
                router_for(method),
                pairs,
                pair_cost_usd=pair_cost_usd,
                tavily=tavily,
            )
        except PairNotCached as err:
            print(f"{case.id} {method}: skipped, {err}", flush=True)
            continue
        if on_diagnosis is not None:
            on_diagnosis(d)
        fixed_hp = None
        if d.status == "confirmed" and not case.healthy:
            fixed_hp = fixed_hyperparameters(case, d.cause)
        row = score(
            case,
            d,
            has_fix=fixed_hp is not None,
            fixed_metric=fixes(fixed_hp) if fixed_hp else None,
            reference=reference,
        )
        rows.append(row)
        print(
            f"{case.id} {method}: {row.status} {row.cause}, top {row.top}, "
            f"fix {row.fix_success}, ${row.cost_usd:.3f}, {row.mini_jobs} jobs",
            flush=True,
        )
    return rows


def fix_trainer(
    tf: TokenFactory, storage: LocalStorage, case: BenchCase, prefix: str
) -> Callable[[dict], tuple[float, str]]:
    """Train fixed hyperparameters on the case's whole training file, score held-out."""
    from jobs.eval_local.exact_match import evaluate_adapter

    def train(hyperparameters: dict) -> tuple[float, str]:
        digest = hashlib.sha1(fix_key(hyperparameters).encode()).hexdigest()[:10]
        result = ReferenceRun(
            tf=tf,
            storage=storage,
            model=case.model,
            train_path=Path(case.train),
            valid_path=Path(case.valid),
            heldout_path=Path(case.heldout),
            score=evaluate_adapter,
            prefix=f"{prefix}/fix-{digest}",
            grid={"fix": {}},
            base_hp=hyperparameters,
        )()
        return result["heldout_exact_match"], result["job_id"]

    return train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "heldout"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--cases", nargs="+", help="only these case IDs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    storage = LocalStorage("artifacts")
    tf = TokenFactory.from_settings()
    tavily = Tavily.from_settings(storage) if get_settings().tavily_api_key else None
    rows: list[Row] = []
    for case in load_cases(args.split):
        if args.cases and case.id not in args.cases:
            continue
        prefix = f"bench/{args.split}/{case.id}"
        pairs = PairCache(
            storage,
            f"{prefix}/pairs.json",
            None
            if args.no_train
            else tf_pair_runner(tf, storage, case, args.seeds, f"{prefix}/tf"),
        )
        fixes = FixCache(
            storage,
            f"{prefix}/fixes.json",
            None if args.no_train else fix_trainer(tf, storage, case, prefix),
        )
        for report in case.reports:
            seed_cache(pairs, Path(report))
            seed_fixes(fixes, Path(report))
        rows += run_case(
            case,
            args.methods,
            router_for=lambda m, c=case: Router(tf, CallLog(f"bench-{c.id}-{m}")),
            pairs=pairs,
            fixes=fixes,
            reference=reference_metric(case.task, case.model),
            pair_cost_usd=pair_cost(case, len(args.seeds)),
            tavily=tavily,
            on_diagnosis=lambda d, p=prefix: storage.put_text(
                f"{p}/{d.method}.json", json.dumps(asdict(d), indent=2)
            ),
        )

    out = f"bench/{args.split}"
    storage.put_text(
        f"{out}/rows.jsonl", "".join(json.dumps(asdict(r)) + "\n" for r in rows)
    )
    summary = table(summarize(rows))
    storage.put_text(f"{out}/summary.md", summary + "\n")
    print(summary)
    print(f"Rows: artifacts/{out}/rows.jsonl")


if __name__ == "__main__":
    main()
