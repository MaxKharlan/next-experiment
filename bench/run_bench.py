"""The benchmark: every method on every case of a split, scored as in section 10.

    uv run --group ml --group eval python -m bench.run_bench --split dev --no-train

For each case, pairs and fixes are cached under artifacts/bench/<split>/<case id>/,
so a rerun and the other methods reuse them; the investigations listed in the
case's "reports" seed both caches. Pairs are scored on the case's validation file,
or on --check-set: then their cache is pairs-<set name>.json, which
scripts/rescore_pairs.py fills from the saved adapters without training. --noise-sd
sets the noise floor of every causal check, measured by that script, and
--min-effect the smallest effect that counts as a cause, about 3 noise SDs. A pair or a fix that is not cached trains on
Token Factory, unless --no-train: then a method that needs a missing pair skips the
case, and a missing fix leaves fix success unknown. Diagnoses go next to the caches,
the rows to artifacts/bench/<split>/rows.jsonl and the table to summary.md.

--resume continues a run that stopped: a method whose diagnosis is already saved for
a case is scored from that file instead of running again, so the model is not asked
twice and the finished cases keep the answers they gave.
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
    seed_reference,
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
    noise_sd: float | None = None,
    min_effect: float | None = None,
    saved: Callable[[str], Diagnosis | None] | None = None,
) -> list[Row]:
    rows = []
    for method in methods:
        d = saved(method) if saved is not None else None
        if d is None:
            try:
                d = run_method(
                    method,
                    case,
                    router_for(method),
                    pairs,
                    pair_cost_usd=pair_cost_usd,
                    tavily=tavily,
                    noise_sd=noise_sd,
                    min_effect=min_effect,
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


# a method's diagnosis is saved under the name it gives itself
SAVED_AS = {"react": "react_super"}


def saved_diagnosis(
    storage: LocalStorage, prefix: str, method: str
) -> Diagnosis | None:
    """The diagnosis a method already gave on a case, if it was saved."""
    key = f"{prefix}/{SAVED_AS.get(method, method)}.json"
    if not storage.exists(key):
        return None
    data = json.loads(storage.get_text(key))
    data["ranking"] = [tuple(pair) for pair in data["ranking"]]
    return Diagnosis(**data)


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
    parser.add_argument("--check-set", help="score pairs on this file, not validation")
    parser.add_argument("--noise-sd", type=float, help="noise floor of the checks")
    parser.add_argument("--min-effect", type=float, help="smallest effect of a cause")
    parser.add_argument(
        "--resume", action="store_true", help="score saved diagnoses again"
    )
    args = parser.parse_args()
    pairs_name = f"pairs-{Path(args.check_set).stem}" if args.check_set else "pairs"

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
            f"{prefix}/{pairs_name}.json",
            None
            if args.no_train
            else tf_pair_runner(
                tf, storage, case, args.seeds, f"{prefix}/tf", args.check_set
            ),
        )
        fixes = FixCache(
            storage,
            f"{prefix}/fixes.json",
            None if args.no_train else fix_trainer(tf, storage, case, prefix),
        )
        for report in case.reports:
            if not args.check_set:  # the reports' pairs were scored on validation
                seed_cache(pairs, Path(report))
            seed_fixes(fixes, Path(report))
        seed_reference(fixes, case)
        rows += run_case(
            case,
            args.methods,
            router_for=lambda m, c=case: Router(tf, CallLog(f"bench-{c.id}-{m}")),
            pairs=pairs,
            fixes=fixes,
            reference=reference_metric(case.task, case.model),
            pair_cost_usd=pair_cost(case, len(args.seeds)),
            tavily=tavily,
            noise_sd=args.noise_sd,
            min_effect=args.min_effect,
            on_diagnosis=lambda d, p=prefix: storage.put_text(
                f"{p}/{d.method}.json", json.dumps(asdict(d), indent=2)
            ),
            saved=(lambda m, p=prefix: saved_diagnosis(storage, p, m))
            if args.resume
            else None,
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
