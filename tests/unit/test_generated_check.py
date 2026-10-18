import json
import os

import pytest

from core.agent.sandbox_check import CheckCode, generated_check
from jobs.sandbox.runner import SandboxResult, SandboxUnavailable


class Router:
    def __init__(self, code):
        self.code = code
        self.calls = []

    def structured(self, step, messages, schema, **kwargs):
        self.calls.append((step, messages, kwargs))
        return CheckCode(lines=self.code.splitlines())


def test_generation_passes_only_explicit_input_and_executes_generated_code():
    router = Router("print(json.dumps({'count': len(DATA['rows'])}))")
    captured = []

    def execute(source, **limits):
        captured.append((source, limits))
        return SandboxResult(0, '{"count":2}', "", False, False)

    result = generated_check(
        router, {"question": "Count rows", "data": {"rows": [1, 2]}}, executor=execute
    )
    assert result["status"] == "completed"
    assert result["observations"] == {"count": 2}
    assert router.calls[0][0] == "probe_code"
    assert captured[0][1] == {"timeout": 10, "max_output_bytes": 8192}
    assert result["causal_confirmation"] is False
    assert "data" not in result


@pytest.mark.parametrize(
    "result,status",
    [
        (SandboxResult(1, "", "error", False, False), "execution_failed"),
        (SandboxResult(0, "{}", "", True, False), "execution_failed"),
        (SandboxResult(0, "{}", "", False, True), "execution_failed"),
        (SandboxResult(0, "[]", "", False, False), "invalid_output"),
        (SandboxResult(0, '{"v":NaN}', "", False, False), "invalid_output"),
        (SandboxResult(0, '{"v":1e400}', "", False, False), "invalid_output"),
    ],
)
def test_failure_never_becomes_valid_evidence(result, status):
    report = generated_check(
        Router("pass"),
        {"question": "check", "data": {}},
        executor=lambda *a, **kw: result,
    )
    assert report["status"] == status
    assert "observations" not in report


def test_unavailable_has_no_host_fallback():
    def unavailable(*a, **kw):
        raise SandboxUnavailable("offline")

    assert (
        generated_check(
            Router("pass"), {"question": "check", "data": {}}, executor=unavailable
        )["status"]
        == "unavailable"
    )


def test_oversized_data_rejected_before_model_call():
    router = Router("pass")
    with pytest.raises(ValueError):
        generated_check(router, {"question": "check", "data": {"x": "x" * 16001}})
    assert router.calls == []


@pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_SANDBOX_TESTS") != "1", reason="requires Docker"
)
def test_generated_code_runs_in_real_docker_without_host_secret(monkeypatch):
    monkeypatch.setenv("TOKEN_FACTORY_API_KEY", "host-secret")
    report = generated_check(
        Router(
            "import os\nprint(json.dumps({'sum':sum(DATA['values']), 'secret':os.getenv('TOKEN_FACTORY_API_KEY')}))"
        ),
        {"question": "Sum values", "data": {"values": [2, 3]}},
    )
    assert report["status"] == "completed"
    assert report["observations"] == {"sum": 5, "secret": None}
    json.dumps(report, allow_nan=False)


def test_html_instead_of_python_is_rejected_by_structured_schema():
    with pytest.raises(ValueError, match="valid Python"):
        CheckCode(lines=["import json<br>print({})"])
