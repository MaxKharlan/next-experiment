"""B22: what runs an investigation in the background: a replay or the live agent.

replay: streams a saved investigation from demo/ step by step. The public demo uses
only this, because it has no GPU and spends no money: on the server with
PUBLIC_DEMO=true, and as a static page on GitHub Pages that plays the same events in
the browser (scripts/build_pages.py).
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

from apps.api.live_protocol import (
    estimate_pair_cost,
    prepare_live_run,
    verify_datasets,
)
from apps.api.store import InvestigationStore

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
RETRAIN_KINDS = {"retrain_proposal", "retrain", "comparison", "retrain_declined"}
REPLAY_DELAY_S = 1.5  # pause before each replayed step


def demo_cases() -> dict[str, dict[str, Any]]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(DEMO_DIR.glob("*.json"))
    }


def demo_index() -> list[dict[str, str]]:
    """Name, title and note of each saved investigation, for the case picker."""
    return [
        {
            "name": name,
            "title": d.get("title", name),
            "note": d.get("note", ""),
            "status": d["report"]["status"],
            "category": "Measured experiments"
            if d["report"].get("measured_experiment")
            else "Local audits"
            if d["report"].get("audit_only")
            else "Agent investigations"
            if d["report"]["status"] == "confirmed"
            else "Unconfirmed traces",
        }
        for name, d in demo_cases().items()
    ]


def replay(store: InvestigationStore, inv_id: str, name: str, delay_s: float) -> None:
    for pause, kind, data in replay_events(name, delay_s):
        time.sleep(pause)
        store.emit(inv_id, kind, data)


def replay_events(
    name: str, delay_s: float = REPLAY_DELAY_S
) -> list[tuple[float, str, dict[str, Any]]]:
    """A replay as (pause before the event, event type, data).

    The server emits these one by one; the static demo plays the same list.
    """
    saved = demo_cases()[name]
    report = saved["report"]
    return [
        (0.0, "started", {"case": saved["case"], "replay_of": name}),
        *((delay_s, "step", step) for step in replay_steps(report)),
        (0.0, "finished", _summary(report)),
    ]


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


def live(
    store: InvestigationStore,
    inv_id: str,
    backend: str,
    *,
    prepared: dict | None = None,
    resume_state: dict | None = None,
    approved_paid_step: dict | None = None,
) -> None:
    """Run an explicit case or the existing demo case."""
    from apps.api.case_input import verify_original_job
    from bench.cases import load_cases
    from core.agent.context import InvestigationContext
    from core.agent.loop import PaidStepApprovalRequired
    from core.agent.router import Router
    from core.config import get_settings
    from core.connectors.storage import LocalStorage
    from core.connectors.tavily import Tavily
    from core.connectors.token_factory import TokenFactory
    from core.interventions.answer_check import investigate_with_answers
    from core.probes.weight_cache import investigation_probes, weights_enabled
    from core.telemetry.calls import CallLog
    from scripts.smoke_agent_steps import CASE, JOB_HP

    spending = None
    try:
        storage = LocalStorage("artifacts")
        tf = TokenFactory.from_settings()
        router = Router(
            tf,
            CallLog(
                inv_id,
                on_add=lambda call: store.emit(inv_id, "llm_call", asdict(call)),
            ),
        )
        tavily = (
            Tavily.from_settings(storage) if get_settings().tavily_api_key else None
        )
        protocol = None
        case_data, job_hp, job_id = CASE, JOB_HP, "local"
        if prepared is not None:
            if backend != "tf":
                raise ValueError("Explicit cases require the tf backend")
            case_data = prepared["agent_case"]
            protocol = prepared["protocol"]
            job_hp = protocol["hyperparameters"]
            job_id = protocol["job_id"]
            verify_datasets(protocol)
            if resume_state is None:
                verify_original_job(tf, prepared)
        elif backend == "tf":
            source = next(
                case for case in load_cases("dev") if case.id == "fault6-lr1e-5"
            )
            case_data = source.case
            job_hp = source.hyperparameters
            job_id = source.job_id
            protocol = prepare_live_run(source)
        elif weights_enabled():
            raise ValueError("B25 is calibrated for Token Factory; use the tf backend")
        ctx = InvestigationContext(
            investigation_id=inv_id,
            mode="A",
            job_id=job_id,
            job_config={
                "model": case_data["job"]["model"],
                "hyperparameters": job_hp,
                "checkpoints": case_data["checkpoints"],
            },
            budget_left_usd=protocol["budget_usd"] if protocol else 1.0,
            files=(
                {role: entry["path"] for role, entry in protocol["datasets"].items()}
                if protocol
                else {}
            ),
        )
        from core.db.spending import SpendingLedger

        spending = SpendingLedger(
            Path("artifacts/investigations") / inv_id / "spending.sqlite3",
            ctx.budget_left_usd,
        )
        tf.spending = spending
        if resume_state is None:
            store.emit(
                inv_id,
                "started",
                {
                    "case": case_data,
                    "backend": backend,
                    "job_id": job_id,
                    "protocol": protocol,
                },
            )
        approval_consumed = False

        def approve_paid_step(proposal):
            nonlocal approval_consumed
            if approved_paid_step is not None and not approval_consumed:
                expected = approved_paid_step["proposal"]
                if (
                    proposal["kind"] != expected["kind"]
                    or proposal["estimated_cost_usd"] != expected["estimated_cost_usd"]
                ):
                    raise ValueError("The resumed paid step differs from its proposal")
                approval_consumed = True
                return True
            raise PaidStepApprovalRequired(proposal)

        result = investigate_with_answers(
            ctx,
            case_data,
            router,
            _pair_runner(backend, inv_id, tf, storage, job_hp, protocol),
            probes=investigation_probes(ctx),
            spending=spending,
            approve_paid_step=approve_paid_step,
            resume_state=resume_state,
            hypothesis_samples=getattr(get_settings(), "hypothesis_samples", 1),
            tavily=tavily,
            on_step=lambda step: store.emit(inv_id, "step", step),
            propose_retrain=_proposal(backend, protocol),
            **(
                {
                    "noise_sd": protocol["noise_sd"],
                    "min_effect": protocol["min_effect"],
                    "intervention_cost_usd": protocol["pair_estimate_usd"],
                    "estimate_pair_cost": lambda target, control: estimate_pair_cost(
                        protocol, target, control
                    ),
                }
                if protocol
                else {}
            ),
        )
        result.protocol = protocol
        result.save(storage)
        summary = _summary(json.loads(result.to_json()))
        summary["spending"] = spending.snapshot()
        if result.retrain_plan:
            plan = result.retrain_plan
            store.emit(
                inv_id,
                "awaiting_approval",
                {"plan": plan, "summary": summary, "protocol": protocol},
            )
        else:
            store.emit(inv_id, "finished", summary)
    except PaidStepApprovalRequired as pause:
        public_proposal = {
            key: value for key, value in pause.proposal.items() if key != "continuation"
        }
        store.emit(
            inv_id,
            "awaiting_paid_step",
            {
                "proposal": public_proposal,
                "continuation": pause.proposal["continuation"],
                "prepared": prepared,
                "spending": spending.snapshot(),
            },
        )
    except Exception as err:  # noqa: BLE001 - the stream must end on any failure
        _fail(store, inv_id, err, spending=spending)


def resume_paid(
    store: InvestigationStore,
    inv_id: str,
    waiting: dict,
    approved: bool,
) -> None:
    """Resume a saved investigation at the exact paid operation it proposed."""
    if not approved:
        store.emit(
            inv_id,
            "finished",
            {
                "status": "inconclusive",
                "reason": "paid step declined",
                "spending": waiting["spending"],
            },
        )
        return
    live(
        store,
        inv_id,
        "tf",
        prepared=waiting.get("prepared"),
        resume_state=waiting["continuation"],
        approved_paid_step=waiting,
    )


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
            protocol = waiting.get("protocol")
            dataset_args = {}
            if protocol is not None:
                verify_datasets(protocol)
                datasets = protocol["datasets"]
                dataset_args = {
                    "train": Path(datasets["train"]["path"]),
                    "valid": Path(datasets["validation"]["path"]),
                    "heldout": Path(datasets["test"]["path"]),
                    "evaluator_spec": protocol.get("evaluator"),
                }
            retrainer = retrain_tf.make_retrainer(
                TokenFactory.from_settings(),
                LocalStorage("artifacts"),
                inv_id,
                **dataset_args,
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


def _fail(
    store: InvestigationStore, inv_id: str, err: Exception, *, spending=None
) -> None:
    store.emit(
        inv_id,
        "failed",
        {
            "error": f"{type(err).__name__}: {err}",
            "trace": traceback.format_exc(),
            **({"spending": spending.snapshot()} if spending is not None else {}),
        },
    )


def _proposal(backend: str, protocol: dict | None = None):
    """The full retrain runs on Token Factory, so only the tf backend proposes it."""
    if backend != "tf":
        return None
    if protocol is not None:
        from core.interventions.retrain import plan_retrain

        return lambda target, control: asdict(
            plan_retrain(
                target,
                protocol["hyperparameters"],
                protocol["model"],
                protocol["datasets"]["train"]["examples"],
                protocol["tokens_per_example"],
                protocol["job_id"],
            )
        )
    from scripts.retrain_tf import propose

    return lambda target, control: asdict(propose(target, control))


def _pair_runner(
    backend: str,
    inv_id: str,
    tf: Any,
    storage: Any,
    job_hp: dict,
    protocol: dict | None = None,
):
    train, valid = Path("artifacts/b6/train.jsonl"), Path("artifacts/b6/valid.jsonl")
    if backend == "tf":
        if protocol is None:
            raise ValueError("The TF runner requires a validated protocol")
        datasets = protocol["datasets"]
        train = Path(datasets["train"]["path"])
        valid = Path(datasets["validation"]["path"])
        check = Path(datasets["check"]["path"])
        from core.case import EvaluatorSpec
        from core.eval import evaluate
        from core.interventions.answer_check import DataAwareTFPairRunner

        def run_tf(target, control):
            verify_datasets(protocol)
            DataAwareTFPairRunner(
                protocol=protocol,
                tf=tf,
                storage=storage,
                model=protocol["model"],
                train_path=train,
                valid_path=valid,
                base_hp=job_hp,
                evaluate=lambda adapter: (
                    evaluate(
                        check,
                        EvaluatorSpec.model_validate(
                            protocol.get("evaluator")
                            or {"name": "intent_label", "metric": "exact_match"}
                        ),
                        adapter=adapter,
                    ).value
                ),
                seeds=protocol["seeds"],
                prefix=f"investigations/{inv_id}/tf/{target.hypothesis_id}",
            )(target, control)

        run_tf.supports_answer_drop = True
        return run_tf

    from jobs.kaggle.run_pair import train_arm

    def run_local(target, control):
        out = Path("artifacts/investigations") / inv_id / "runs" / target.hypothesis_id
        for arm in (target, control):
            train_arm(arm, job_hp, train, valid, out, [0, 1])

    return run_local


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    training = sum(
        step.get("cost_usd", 0.0)
        for step in report.get("steps", [])
        if step["kind"] == "intervention"
    )
    measurement = report.get("measured_experiment")
    if measurement:
        training = measurement["estimated_training_cost_usd"]
    return {
        "measured_experiment": measurement,
        "estimated_training_cost_usd": training,
        "estimated_total_cost_usd": (
            training + report.get("llm_cost_usd", 0.0) if training is not None else None
        ),
        "status": report["status"],
        "audit_only": report.get("audit_only", False),
        "evidence_scope": report.get("evidence_scope"),
        "llm_cost_usd": report.get("llm_cost_usd", 0.0),
        "llm_calls": report.get("llm_calls", {}),
    }
