"""B22: what runs an investigation in the background: a replay or the live agent.

replay: streams a saved investigation from demo/ step by step. The public demo uses
only this, because it has no GPU and spends no money.
live: runs the agent on the fault-6 case. Causal checks train locally ("local") or
as Token Factory jobs ("tf"); it needs the keys in .env and the ml and eval groups.
On Token Factory a confirmed fix ends with the proposal of the full retrain, and the
investigation waits for the user's answer.
retrain: the answer. It trains the fix and compares three models, or finishes
without them.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from apps.api.store import InvestigationStore

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
RETRAIN_KINDS = {"retrain_proposal", "retrain", "comparison", "retrain_declined"}


def demo_cases() -> dict[str, dict[str, Any]]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(DEMO_DIR.glob("*.json"))
    }


def replay(store: InvestigationStore, inv_id: str, name: str, delay_s: float) -> None:
    saved = demo_cases()[name]
    report = saved["report"]
    store.emit(inv_id, "started", {"case": saved["case"], "replay_of": name})
    for step in replay_steps(report):
        time.sleep(delay_s)
        store.emit(inv_id, "step", step)
    store.emit(inv_id, "finished", _summary(report))


def replay_steps(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The steps of a saved report in the order of the screen.

    Runs saved before the loop recorded the verdict as a step get it from the
    report, placed before the retrain steps that follow it.
    """
    steps = [
        {**s, "hypotheses": report["hypotheses"]}
        if s["kind"] == "hypotheses" and "hypotheses" not in s
        else s
        for s in report["steps"]
    ]
    if report.get("verdict") and not any(s["kind"] == "verdict" for s in steps):
        at = next(
            (i for i, s in enumerate(steps) if s["kind"] in RETRAIN_KINDS), len(steps)
        )
        steps.insert(at, {"kind": "verdict", **report["verdict"]})
    return steps


def live(store: InvestigationStore, inv_id: str, backend: str) -> None:
    """The agent on the fault-6 case; heavy imports stay inside."""
    from core.agent.context import InvestigationContext
    from core.agent.loop import investigate
    from core.agent.router import Router
    from core.config import get_settings
    from core.connectors.storage import LocalStorage
    from core.connectors.tavily import Tavily
    from core.connectors.token_factory import TokenFactory
    from core.telemetry.calls import CallLog
    from scripts.smoke_agent_steps import CASE, JOB_HP

    try:
        storage = LocalStorage("artifacts")
        tf = TokenFactory.from_settings()
        router = Router(tf, CallLog(inv_id))
        tavily = (
            Tavily.from_settings(storage) if get_settings().tavily_api_key else None
        )
        ctx = InvestigationContext(
            investigation_id=inv_id,
            mode="A",
            job_id="local",
            job_config={"hyperparameters": JOB_HP, "checkpoints": CASE["checkpoints"]},
            budget_left_usd=1.0,
        )
        store.emit(inv_id, "started", {"case": CASE, "backend": backend})
        result = investigate(
            ctx,
            CASE,
            router,
            _pair_runner(backend, inv_id, tf, storage, JOB_HP),
            tavily=tavily,
            on_step=lambda step: store.emit(inv_id, "step", step),
            propose_retrain=_proposal(backend),
        )
        result.save(storage)
        summary = _summary(json.loads(result.to_json()))
        if result.retrain_plan:
            plan = result.retrain_plan
            store.emit(inv_id, "awaiting_approval", {"plan": plan, "summary": summary})
        else:
            store.emit(inv_id, "finished", summary)
    except Exception as err:  # noqa: BLE001 - the stream must end on any failure
        _fail(store, inv_id, err)


def retrain(
    store: InvestigationStore, inv_id: str, waiting: dict, approved: bool
) -> None:
    """The user's answer to the proposed retrain: train and compare, or finish."""
    from core.connectors.storage import LocalStorage
    from core.connectors.token_factory import TokenFactory
    from core.interventions.retrain import RetrainPlan
    from scripts import retrain_tf

    try:
        if approved:
            retrainer = retrain_tf.make_retrainer(
                TokenFactory.from_settings(), LocalStorage("artifacts"), inv_id
            )
            comparison = retrainer(
                RetrainPlan(**waiting["plan"]),
                on_step=lambda step: store.emit(inv_id, "step", step),
            )
            store.emit(inv_id, "step", comparison)
        else:
            store.emit(inv_id, "step", {"kind": "retrain_declined"})
        store.emit(inv_id, "finished", waiting["summary"])
    except Exception as err:  # noqa: BLE001 - the stream must end on any failure
        _fail(store, inv_id, err)


def _fail(store: InvestigationStore, inv_id: str, err: Exception) -> None:
    store.emit(
        inv_id,
        "failed",
        {"error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()},
    )


def _proposal(backend: str):
    """The full retrain runs on Token Factory, so only the tf backend proposes it."""
    if backend != "tf":
        return None
    from scripts.retrain_tf import propose

    return lambda target, control: asdict(propose(target, control))


def _pair_runner(backend: str, inv_id: str, tf: Any, storage: Any, job_hp: dict):
    train, valid = Path("artifacts/b6/train.jsonl"), Path("artifacts/b6/valid.jsonl")
    if backend == "tf":
        from core.interventions.tf_runner import TokenFactoryPairRunner
        from jobs.eval_local.exact_match import evaluate_adapter

        def run_tf(target, control):
            TokenFactoryPairRunner(
                tf=tf,
                storage=storage,
                model="Qwen/Qwen3-0.6B",
                train_path=train,
                valid_path=valid,
                base_hp=job_hp,
                evaluate=lambda adapter: evaluate_adapter(adapter, valid),
                prefix=f"investigations/{inv_id}/tf/{target.hypothesis_id}",
            )(target, control)

        return run_tf

    from jobs.kaggle.run_pair import train_arm

    def run_local(target, control):
        out = Path("artifacts/investigations") / inv_id / "runs" / target.hypothesis_id
        for arm in (target, control):
            train_arm(arm, job_hp, train, valid, out, [0, 1])

    return run_local


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "llm_cost_usd": report.get("llm_cost_usd", 0.0),
        "llm_calls": report.get("llm_calls", {}),
    }
