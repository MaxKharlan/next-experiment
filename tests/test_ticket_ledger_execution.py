import copy
import json
from types import SimpleNamespace

import pytest

from bench.tasks.ticket_ledger import ARMS, FAMILIES
from core.db.spending import SpendingLedger
from scripts import execute_ticket_ledger as execution
from scripts.execute_ticket_ledger_v2 import check_accepted
from scripts.prepare_ticket_ledger import PROTOCOL


def result():
    truth = json.dumps(
        {
            "ticket_id": "T-800000",
            "product": "router",
            "priority": "normal",
            "callback": False,
            "due_date": None,
        }
    )
    return {"expected": truth, "answer": truth, "hit_limit": False}


def test_limit_hit_and_malformed_answer_remain_in_primary_denominator():
    row = result()
    assert execution.metrics(row)["strict_exact"] == 1
    row["hit_limit"] = True
    assert execution.metrics(row)["strict_exact"] == 0
    row.update(hit_limit=False, answer="not JSON")
    assert execution.metrics(row) == {
        "strict_exact": 0.0,
        "schema_valid": 0.0,
        "field_accuracy": 0.0,
    }


def passing_scores():
    scores, weights = {}, {}
    for arm in ARMS:
        for seed in PROTOCOL["order_seeds"]:
            name = f"{arm}-s{seed}"
            history = 0.7 if arm == "healthy" else 0.4
            scores[name] = {
                "serialization": {"strict_exact": 1.0},
                **{
                    group: {"strict_exact": history} for group in ("history", *FAMILIES)
                },
            }
            weights[name] = name
    return scores, weights


def test_all_gates_and_distinct_weights_are_required():
    scores, weights = passing_scores()
    assert execution.gate_failures(scores, weights, PROTOCOL)[0] == []
    for seed in PROTOCOL["order_seeds"]:
        weights[f"healthy-s{seed}"] = "same"
        scores[f"healthy-s{seed}"]["copy"]["strict_exact"] = 0.2
    failures, _ = execution.gate_failures(scores, weights, PROTOCOL)
    assert failures == ["healthy_copy_floor", "healthy_distinct_weights"]
    scores.pop(next(iter(scores)))
    with pytest.raises(ValueError, match="All arm"):
        execution.gate_failures(scores, weights, PROTOCOL)


def test_ceiling_and_control_gaps_cannot_be_replaced_by_format_success():
    scores, weights = passing_scores()
    for seed in PROTOCOL["order_seeds"]:
        scores[f"healthy-s{seed}"]["history"]["strict_exact"] = 1.0
        scores[f"cut-s{seed}"]["history"]["strict_exact"] = 0.98
    failures, _ = execution.gate_failures(scores, weights, PROTOCOL)
    assert "healthy_history_range" in failures
    assert "healthy_minus_cut" in failures


def test_tampered_completed_row_is_rejected():
    row = {
        **result(),
        "line": 1,
        "source": "copy",
        "stopped_on_eos": True,
    }
    row.update(execution.metrics(row))
    record = {"protocol_sha256": "bound", "rows": [row], "complete": False}
    examples = [SimpleNamespace(line=1, source="copy", expected=row["expected"])]
    execution.checked_rows(record, examples, {"protocol_sha256": "bound"})
    damaged = copy.deepcopy(record)
    damaged["rows"][0]["answer"] = "{}"
    with pytest.raises(ValueError, match="Metric drift"):
        execution.checked_rows(damaged, examples, {"protocol_sha256": "bound"})


def test_unknown_submission_retains_hold_and_cannot_submit_again(tmp_path, monkeypatch):
    ledger_path = tmp_path / "budget.sqlite3"
    plan = {
        "id": "test",
        "spending_ledger": str(ledger_path),
        "model": "model",
        "validation": "validation",
        "recipe": {},
        "jobs": {"healthy-s1": {"reservation_usd": 1.0, "train": "train", "seed": 1}},
    }
    state = {}
    monkeypatch.setattr(execution, "load", lambda root: (plan, state, "digest"))
    calls = []

    def ambiguous(*args, **kwargs):
        calls.append("create")
        raise TimeoutError("unknown paid outcome")

    tf = SimpleNamespace(upload=lambda path: "file", create_job=ambiguous)
    monkeypatch.setattr(execution, "provider", lambda env: tf)
    with pytest.raises(TimeoutError):
        execution.submit(tmp_path, None)
    snapshot = SpendingLedger(ledger_path, 25).snapshot()
    assert snapshot["committed_usd"] == 1.0
    assert snapshot["operations"][0]["status"] == "pending"
    with pytest.raises(ValueError, match="Ambiguous"):
        execution.submit(tmp_path, None)
    assert calls == ["create"]


@pytest.mark.parametrize("changed", ["lora", "seed", "training_file", "learning_rate"])
def test_provider_echo_must_match_planned_lora_mode_and_inputs(changed):
    kwargs = {
        "hyperparameters": {"lora": True, "learning_rate": 0.0001},
        "validation_file": "valid",
        "seed": 2401,
    }
    observed = {
        "model": "model",
        "training_file": "train",
        "validation_file": "valid",
        "seed": 2401,
        "hyperparameters": dict(kwargs["hyperparameters"]),
    }
    assert (
        check_accepted(
            SimpleNamespace(model_dump=lambda: observed), "model", "train", kwargs
        )
        == observed
    )
    if changed == "lora":
        observed["hyperparameters"]["lora"] = False
    elif changed == "learning_rate":
        observed["hyperparameters"]["learning_rate"] = 0.01
    elif changed == "seed":
        observed["seed"] = 7
    else:
        observed["training_file"] = "other"
    with pytest.raises(ValueError, match="Provider"):
        check_accepted(
            SimpleNamespace(model_dump=lambda: observed), "model", "train", kwargs
        )
