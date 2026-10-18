"""Build a downloadable investigation snapshot from persisted API events."""

from __future__ import annotations

from typing import Any


def build_report(snapshot: dict[str, Any]) -> dict[str, Any]:
    meta, events = snapshot["meta"], snapshot["events"]
    steps = [event["data"] for event in events if event["type"] == "step"]

    def last_event(kind):
        return next((e["data"] for e in reversed(events) if e["type"] == kind), None)

    def last_step(kind):
        return next(
            (step for step in reversed(steps) if step.get("kind") == kind), None
        )

    calls = [event["data"] for event in events if event["type"] == "llm_call"]
    usage = None
    if calls:
        usage = {
            "calls": calls,
            "call_count": len(calls),
            "tokens_in": sum(call["tokens_in"] for call in calls),
            "tokens_out": sum(call["tokens_out"] for call in calls),
            "cost_usd": sum(call["cost_usd"] for call in calls),
            "scope": "Recorded LLM responses only; excludes training and requests without usage data",
        }
    started = last_event("started") or {}
    waiting = last_event("awaiting_approval") or {}
    summary = last_event("finished") or waiting.get("summary") or {}
    verdict = last_step("verdict")
    error = last_event("failed")
    if meta["status"] == "failed" and error is None:
        error = {"error": "The stored investigation was interrupted before completion"}
    hypotheses = last_step("hypotheses") or {}
    comparison = last_step("comparison")
    return {
        "version": 1,
        "source": "persisted_api_events",
        "investigation_id": meta["id"],
        "title": meta.get("title", ""),
        "backend": meta["backend"],
        "created_at": meta["created_at"],
        "execution_status": meta["status"],
        "partial": meta["status"] != "finished",
        "diagnosis_status": verdict.get("status") if verdict else summary.get("status"),
        "case": started.get("case", meta.get("case", {})),
        "input_case": meta.get("case", {}),
        "protocol": started.get("protocol") or waiting.get("protocol"),
        "hypotheses": hypotheses.get("hypotheses", []),
        "steps": steps,
        "verdict": verdict,
        "retrain_plan": waiting.get("plan") or last_step("retrain_proposal"),
        "approval": last_event("resumed"),
        "comparison": comparison,
        "summary": summary,
        "llm_usage": usage,
        "error": error,
        "events": events,
        "last_event_seq": events[-1]["seq"] if events else None,
        "cost_scope": "Summary costs are recorded as emitted; full retrain costs may be absent",
    }
