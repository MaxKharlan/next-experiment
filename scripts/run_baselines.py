"""B19: the agent and the comparison methods on the fault-6 case of the demo.

- single_ultra: one Ultra call with the whole case and no training, about one cent.
- react: Super picks probes and causal checks itself, without the planner.
- checklist: every level-0 probe, then a causal check for each recipe fault.
- next_experiment: the agent, as in scripts/investigate_tf.py.

Causal checks go through a cache that all methods share
(artifacts/bench/fault6/pairs.json). It starts with the pairs of the 2026-10-16
investigation, read from its report. A pair that is not in the cache trains on
Token Factory: 4 Qwen3-0.6B jobs, about $0.06. With --no-train such a pair stops
that method instead. Each Diagnosis goes to artifacts/bench/fault6/<method>.json.
For every case of a split with the scores, see bench/run_bench.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bench.cases import BenchCase
from bench.methods import METHODS, pair_cost, run_method, tf_pair_runner
from bench.pair_cache import PairCache, PairNotCached, seed_cache
from core.agent.router import Router
from core.config import get_settings
from core.connectors.storage import LocalStorage
from core.connectors.tavily import Tavily
from core.connectors.token_factory import TokenFactory
from core.telemetry.calls import CallLog
from scripts.smoke_agent_steps import CASE, JOB_HP

FAULT6 = BenchCase(
    id="fault6",
    split="demo",
    task="banking77",
    model="Qwen/Qwen3-0.6B",
    truth=["lr_too_low"],
    hyperparameters=JOB_HP,
    case=CASE,
    train="artifacts/b6/train.jsonl",
    valid="artifacts/b6/valid.jsonl",
    heldout="artifacts/b18/heldout.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument(
        "--report", default="artifacts/investigations/tf-fault6/report.json"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    storage = LocalStorage("artifacts")
    tf = TokenFactory.from_settings()
    cache = PairCache(
        storage,
        "bench/fault6/pairs.json",
        None
        if args.no_train
        else tf_pair_runner(tf, storage, FAULT6, args.seeds, "bench/fault6/tf"),
    )
    print(f"pairs from {args.report}: {seed_cache(cache, Path(args.report))} added")
    cost = pair_cost(FAULT6, len(args.seeds))
    tavily = Tavily.from_settings(storage) if get_settings().tavily_api_key else None

    for method in args.methods:
        router = Router(tf, CallLog(f"bench-fault6-{method}"))
        try:
            d = run_method(
                method, FAULT6, router, cache, pair_cost_usd=cost, tavily=tavily
            )
        except PairNotCached as err:
            print(f"{method}: stopped, {err}; run without --no-train")
            continue
        storage.put_text(f"bench/fault6/{method}.json", json.dumps(asdict(d), indent=2))
        top = ", ".join(f"{c} {p:.2f}" for c, p in d.ranking[:3])
        print(
            f"{method}: {d.status}, cause {d.cause}; top 3: {top}; "
            f"LLM ${d.llm_cost_usd:.4f}, {d.mini_jobs} jobs, "
            f"training about ${d.training_cost_usd:.3f}"
        )


if __name__ == "__main__":
    main()
