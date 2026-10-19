import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api import main
from apps.api.case_input import CaseStart, prepare_case
from apps.api.store import InvestigationStore
from bench.tasks.structured import build, load_case, validate_bundle
from bench.tasks.structured_review import parse_request
from core.case import EvaluatorSpec
from core.connectors.storage import LocalStorage
from core.eval.audit_case import audit_case
from core.eval.metrics import get_metric
from core.eval.structured import structured_summary, validate_answer

TICKET = {
    "ticket_id": "T-123456",
    "product": "router",
    "priority": "high",
    "callback": False,
    "due_date": None,
}
CALL = {
    "name": "cancel_order",
    "arguments": {"order_id": "O-123456", "reason": "duplicate"},
}


@pytest.mark.parametrize(
    "task,truth,metric",
    [
        ("json_extraction", TICKET, "json_fields_exact"),
        ("function_calling", CALL, "function_call_exact"),
    ],
)
def test_structured_equality_ignores_order_not_types(task, truth, metric):
    text = json.dumps(truth)
    assert get_metric(metric)(json.dumps(truth, sort_keys=True, indent=2), text) == 1
    assert get_metric(metric)(text[:-1], text) == 0
    assert get_metric(metric)("```json\n" + text + "\n```", text) == 0
    wrong = deepcopy(truth)
    wrong["extra"] = "unexpected"
    assert get_metric(metric)(json.dumps(wrong), text) == 0
    with pytest.raises(ValueError):
        validate_answer(task, json.dumps(wrong))


@pytest.mark.parametrize("value", [0, "false", None, [], {}])
def test_boolean_is_not_coerced(value):
    wrong = {**TICKET, "callback": value}
    assert get_metric("json_fields_exact")(json.dumps(wrong), json.dumps(TICKET)) == 0


@pytest.mark.parametrize("answer", ['{"a":1,"a":2}', "NaN", "Infinity", "{"])
def test_invalid_json(answer):
    assert get_metric("json_valid")(answer, "{}") == 0


def test_wrong_function_arguments_and_partial_fields():
    wrong = deepcopy(CALL)
    wrong["name"] = "track_order"
    assert get_metric("function_call_exact")(json.dumps(wrong), json.dumps(CALL)) == 0
    changed = {**TICKET, "priority": "low"}
    metrics = structured_summary(
        "json_extraction", [json.dumps(changed)], [json.dumps(TICKET)]
    )
    assert metrics == {
        "exact_match": 0,
        "field_accuracy": 0.8,
        "schema_valid": 1,
        "valid_json": 1,
    }
    changed["due_date"] = "2027-02-30"
    assert get_metric("json_fields_exact")(json.dumps(changed), json.dumps(TICKET)) == 0


def test_generator_is_byte_reproducible_and_validates_all_splits(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build(a)
    build(b)
    for path in a.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (b / path.relative_to(a)).read_bytes()
    for task in ("json_extraction", "function_calling"):
        assert validate_bundle(a / task)["examples"] == 672
        case = load_case(a / task / "audit-case.json")
        report = audit_case(case)
        assert report["status"] == "inconclusive"
        assert all(
            not row["invalid_reference_lines"]
            for row in report["steps"][0]["raw"].values()
        )
        for role in ("train", "validation", "check", "test"):
            for line in (a / task / f"{role}.jsonl").read_text().splitlines():
                row = json.loads(line)
                assert parse_request(task, row["messages"][1]["content"]) == json.loads(
                    row["messages"][-1]["content"]
                )
        (a / task / "check.jsonl").write_text("changed")
        with pytest.raises(ValueError, match="Hash mismatch"):
            validate_bundle(a / task)


@pytest.fixture
def case():
    return load_case(
        Path("datasets/structured-mixed-v1/json_extraction/audit-case.json")
    )


def test_mixed_dataset_has_frozen_super_provenance():
    for task in ("json_extraction", "function_calling"):
        root = Path("datasets/structured-mixed-v1") / task
        assert validate_bundle(root)["examples"] == 696
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["source_counts"]["test"] == {"local": 96, "super": 6}
        assert manifest["super_model"].startswith("nvidia/")
        for role in ("train", "validation", "check", "test"):
            rows = [
                json.loads(line)
                for line in (root / f"{role}.jsonl").read_text().splitlines()
            ]
            for row in rows:
                assert parse_request(task, row["messages"][1]["content"]) == json.loads(
                    row["messages"][-1]["content"]
                )


def test_structured_case_reaches_live_protocol_with_correct_metric(case):
    case.mode = "A"
    case.job.id = "ftjob-test"
    case.job.model = "Qwen/Qwen3-0.6B"
    case.job.hyperparameters = {
        "learning_rate": 0.0002,
        "n_epochs": 3,
        "lora_r": 16,
        "lora_alpha": 16,
        "batch_size": 8,
        "context_length": 8192,
    }
    request = CaseStart(
        case=case,
        experiment={
            "name": "test",
            "noise_sd": 0.02,
            "min_effect": 0.1,
            "tokens_per_example": 300,
        },
    )
    prepared = prepare_case(request)
    assert prepared["protocol"]["evaluator"]["metric"] == "json_fields_exact"
    assert prepared["protocol"]["evaluator"]["parameters"]["max_new_tokens"] == 192
    assert "datasets" not in prepared["agent_case"]


def test_api_audit_persists_and_public_demo_blocks(case, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", InvestigationStore(LocalStorage(tmp_path)))
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    client = TestClient(main.app)
    response = client.post("/api/investigations/audit-case", json=case.model_dump())
    assert response.status_code == 201
    inv = response.json()["id"]
    report = client.get(f"/api/investigations/{inv}/report").json()
    assert report["summary"]["audit_only"] is True
    assert report["diagnosis_status"] == "inconclusive"
    assert report["protocol"]["evaluator"]["metric"] == "json_fields_exact"
    assert "dataset_audit" in client.get(f"/api/investigations/{inv}/events").text
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(public_demo=True))
    assert (
        client.post(
            "/api/investigations/audit-case", json=case.model_dump()
        ).status_code
        == 403
    )
    assert client.get(f"/api/investigations/{inv}/report").status_code == 403


def test_metric_reaches_generation_and_summary(tmp_path, monkeypatch):
    from core.eval import evaluate, generate

    # Test with a loader-free generation stub to isolate metric wiring.
    from core.eval.records import ExampleResult

    text = json.dumps(TICKET)
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "request"},
                    {"role": "assistant", "content": text},
                ]
            }
        )
        + "\n"
    )
    seen = {}

    def answers(model, tok, examples, device, metric, limit):
        seen["limit"] = limit
        return [
            ExampleResult(
                1,
                text,
                json.dumps(TICKET, indent=2),
                metric(json.dumps(TICKET, indent=2), text),
                10,
                30,
                True,
                False,
            )
        ]

    monkeypatch.setattr(
        generate,
        "load_model",
        lambda *a: (SimpleNamespace(to=lambda d: object()), object(), "fake"),
    )
    monkeypatch.setattr(generate, "answers", answers)
    result = evaluate(
        path,
        EvaluatorSpec(name="ticket_json_v1", metric="json_fields_exact"),
        base_model="fake",
        device="cpu",
    )
    assert result.value == 1 and result.details["valid_json"] == 1
    assert seen["limit"] == 192
