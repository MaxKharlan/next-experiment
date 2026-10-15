from types import SimpleNamespace

import httpx
import pytest
from openai import UnprocessableEntityError

from core.agent.schemas import HypothesisList
from core.config import Settings
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import (
    StructuredOutputError,
    TokenFactory,
    extract_json,
)

NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
VALID = (
    '{"hypotheses": [{"id": "h1", "fault_class": "truncation", '
    '"location": "train.jsonl", "prior": 0.6, "rationale": "long answers are cut"}]}'
)


class FakeCompletions:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.texts.pop(0)
        if isinstance(text, Exception):
            raise text
        message = SimpleNamespace(content=text)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _tf(completions=None, jobs=None, files=None):
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        fine_tuning=SimpleNamespace(jobs=jobs),
        files=files,
    )
    return TokenFactory(client, Settings(_env_file=None, token_factory_api_key="t"))


def test_chat_counts_cost():
    call = _tf(FakeCompletions(["ok"])).chat(NANO, [{"role": "user", "content": "hi"}])
    assert call.text == "ok"
    assert call.cost_usd == pytest.approx((1000 * 0.06 + 500 * 0.24) / 1_000_000)


def test_extract_json_strips_reasoning_and_fences():
    text = '<think>long reasoning</think>\n```json\n{"a": 1}\n```'
    assert extract_json(text) == '{"a": 1}'


def test_structured_retries_with_error_feedback():
    fake = FakeCompletions(["not json at all", "\n\n" + VALID])
    result, calls = _tf(fake).structured(
        NANO, [{"role": "user", "content": "x"}], HypothesisList
    )
    assert result.hypotheses[0].fault_class == "truncation"
    assert len(calls) == 2
    assert "does not match the schema" in fake.calls[1]["messages"][-1]["content"]
    response_format = fake.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "HypothesisList"
    assert "hypotheses" in response_format["json_schema"]["schema"]["properties"]


def test_structured_falls_back_when_format_rejected():
    request = httpx.Request("POST", "https://api.tokenfactory.nebius.com/v1/")
    rejected = UnprocessableEntityError(
        "unsupported response_format",
        response=httpx.Response(422, request=request),
        body=None,
    )
    fake = FakeCompletions([rejected, VALID])
    tf = _tf(fake)
    tf.structured(NANO, [{"role": "user", "content": "x"}], HypothesisList)
    assert fake.calls[1]["response_format"] == {"type": "json_object"}
    assert tf.format_level[NANO] == 1


def test_structured_gives_up():
    fake = FakeCompletions(["{}"] * 3)
    with pytest.raises(StructuredOutputError):
        _tf(fake).structured(NANO, [{"role": "user", "content": "x"}], HypothesisList)


def test_wait_polls_until_terminal():
    statuses = iter(["queued", "running", "succeeded"])
    jobs = SimpleNamespace(
        retrieve=lambda job_id: SimpleNamespace(id=job_id, status=next(statuses))
    )
    sleeps = []
    job = _tf(jobs=jobs).wait("ftjob-1", poll_s=5, sleep=sleeps.append)
    assert job.status == "succeeded"
    assert sleeps == [5, 5]


def test_download_checkpoint_to_storage(tmp_path):
    files = SimpleNamespace(
        retrieve=lambda file_id: SimpleNamespace(filename=f"ckpt-1/{file_id}.json"),
        content=lambda file_id: SimpleNamespace(read=lambda: b"{}"),
    )
    checkpoint = SimpleNamespace(id="ckpt-1", result_files=["adapter_config"])
    storage = LocalStorage(tmp_path)
    keys = _tf(files=files).download_checkpoint(
        checkpoint, storage, "inv-1/checkpoints"
    )
    assert keys == ["inv-1/checkpoints/ckpt-1/adapter_config.json"]
    assert storage.get_bytes(keys[0]) == b"{}"
