"""Opt-in generation and isolated execution of a single bounded Python check."""

import ast
import hashlib
import json
import math
from dataclasses import asdict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.agent.steps import load_prompt
from jobs.sandbox.runner import MAX_SOURCE_BYTES, SandboxUnavailable, run_python


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    data: dict


class CheckCode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lines: list[str] = Field(min_length=1, max_length=512)

    @field_validator("lines")
    @classmethod
    def python_syntax(cls, value):
        source = "\n".join(value)
        if len(source.encode()) > 16000:
            raise ValueError("Python code exceeds 16000 bytes")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(
                "Return valid Python source lines, not HTML or Markdown"
            ) from exc
        return value

    @property
    def code(self):
        return "\n".join(self.lines)


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        return _reject_constant(value)
    return number


def generated_check(router, payload: dict, *, executor=run_python) -> dict:
    request = CheckRequest.model_validate(payload)
    serialized = json.dumps(request.data, allow_nan=False)
    if len(serialized.encode()) > 16000:
        raise ValueError("Sandbox input data exceeds 16000 bytes")
    prompt = load_prompt("probe_code")
    code = router.structured(
        "probe_code",
        [
            {"role": "system", "content": prompt.text},
            {
                "role": "user",
                "content": json.dumps(request.model_dump(), allow_nan=False),
            },
        ],
        CheckCode,
        prompt_version=prompt.id,
        max_tokens=4096,
    ).code
    source = "import json\nDATA = json.loads(" + repr(serialized) + ")\n" + code
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise ValueError("Generated check exceeds sandbox source limit")
    record = {
        "question": request.question,
        "code": code,
        "input_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "prompt_version": prompt.id,
        "causal_confirmation": False,
    }
    try:
        result = executor(source, timeout=10, max_output_bytes=8192)
    except SandboxUnavailable as exc:
        return {**record, "status": "unavailable", "error": str(exc)}
    record["execution"] = asdict(result)
    if result.exit_code != 0 or result.timed_out or result.output_limited:
        return {**record, "status": "execution_failed"}
    try:
        output = json.loads(
            result.stdout, parse_constant=_reject_constant, parse_float=_finite_float
        )
        if not isinstance(output, dict):
            return {**record, "status": "invalid_output"}
    except (ValueError, RecursionError):
        return {**record, "status": "invalid_output"}
    return {**record, "status": "completed", "observations": output}
