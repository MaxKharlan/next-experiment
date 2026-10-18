import json

import pytest

from core.agent.context import InvestigationContext
from core.planner.outcome_model import likelihood, load_signatures, outcomes
from core.probes.level0.lengths import (
    TOKEN_FACTORY_CONTEXT_LENGTH,
    ExampleLength,
    count_tokens,
    read_chat_jsonl,
)


class FakeTokenizer:
    """One BOS, two tokens per message header and one token per word of content."""

    name_or_path = "fake/model"
    chat_template = "{{ fake }}"

    def __init__(self, batch_encoding=False):
        self.batch_encoding = batch_encoding
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append((tokenize, add_generation_prompt))
        ids = [0]  # BOS, added once by the template
        for message in messages:
            ids += [1, 2] + [3] * len(message["content"].split())
        return {"input_ids": ids} if self.batch_encoding else ids


def _example(words):
    return {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": " ".join(["w"] * words)},
        ]
    }


def _tokens(words):
    """What FakeTokenizer counts for _example(words): BOS, 2 headers, q, words."""
    return 1 + 2 + 1 + 2 + words


def _write(path, examples):
    path.write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
    return str(path)


def _ctx(files, model="fake/model", **hp):
    return InvestigationContext(
        "inv-1", "A", "ftjob-1", {"model": model, "hyperparameters": hp}, files=files
    )


def _run(ctx, tokenizer=None):
    tokenizer = tokenizer or FakeTokenizer()
    result = ExampleLength(tokenizer=lambda model: tokenizer).run(ctx)
    return result, json.loads(result.summary)


def test_an_example_at_the_limit_fits_and_one_token_more_is_cut(tmp_path):
    limit = _tokens(10)
    train = _write(tmp_path / "train.jsonl", [_example(10)] * 99 + [_example(11)])
    result, raw = _run(_ctx({"train": train}, context_length=limit))
    stats = raw["files"]["train"]
    assert (stats["examples"], stats["over"]) == (100, 1)
    assert stats["max_tokens"] == limit + 1
    assert stats["over_share"] == pytest.approx(0.01)
    assert result.outcome == "truncated"  # 1% reaches the threshold

    train = _write(tmp_path / "fits.jsonl", [_example(10)] * 100)
    result, raw = _run(_ctx({"train": train}, context_length=limit))
    assert raw["files"]["train"]["over"] == 0
    assert result.outcome == "ok"


def test_a_share_below_the_threshold_is_ok(tmp_path):
    train = _write(tmp_path / "train.jsonl", [_example(1)] * 199 + [_example(50)])
    result, raw = _run(_ctx({"train": train}, context_length=_tokens(1)))
    assert raw["files"]["train"]["over"] == 1
    assert result.outcome == "ok"  # 0.5% is below 1%


def test_outcome_rests_on_train_and_validation_is_reported(tmp_path):
    files = {
        "train": _write(tmp_path / "train.jsonl", [_example(2)] * 10),
        "validation": _write(tmp_path / "valid.jsonl", [_example(2), _example(40)]),
    }
    result, raw = _run(_ctx(files, context_length=_tokens(2)))
    assert result.outcome == "ok"
    assert raw["files"]["validation"]["over"] == 1
    assert raw["files"]["validation"]["longest"][0] == [2, _tokens(40)]


def test_raw_result_records_the_limit_tokenizer_and_percentiles(tmp_path):
    train = _write(tmp_path / "train.jsonl", [_example(n) for n in range(1, 101)])
    result, raw = _run(_ctx({"train": train}, context_length=4096))
    assert raw["context_length_source"] == "job"
    assert raw["tokenizer"] == "fake/model"
    assert len(raw["chat_template_sha256"]) == 64
    stats = raw["files"]["train"]
    assert (stats["p50"], stats["p95"], stats["p99"]) == (
        _tokens(50),
        _tokens(95),
        _tokens(99),
    )
    assert [line for line, _ in stats["longest"]] == [100, 99, 98, 97, 96]
    assert result.raw_ref == "investigations/inv-1/probes/example_length.json"
    assert result.cost_usd == 0.0


def test_a_job_without_context_length_uses_the_token_factory_default(tmp_path):
    train = _write(tmp_path / "train.jsonl", [_example(3)])
    _, raw = _run(_ctx({"train": train}))
    assert raw["context_length"] == TOKEN_FACTORY_CONTEXT_LENGTH == 8192
    assert raw["context_length_source"] == "token_factory_default"


@pytest.mark.parametrize("value", [0, -1, 8192.0, "8192", True])
def test_a_bad_context_length_is_an_error(tmp_path, value):
    train = _write(tmp_path / "train.jsonl", [_example(3)])
    with pytest.raises(ValueError, match="context_length"):
        _run(_ctx({"train": train}, context_length=value))


def test_special_tokens_are_counted_once():
    tokenizer = FakeTokenizer()
    assert count_tokens(tokenizer, _example(4)["messages"]) == _tokens(4)
    assert tokenizer.calls == [(True, False)]  # tokenized by the template itself
    encoded = FakeTokenizer(batch_encoding=True)
    assert count_tokens(encoded, _example(4)["messages"]) == _tokens(4)


def test_available_needs_the_model_and_a_local_train_file(tmp_path):
    train = _write(tmp_path / "train.jsonl", [_example(1)])
    probe = ExampleLength(tokenizer=lambda model: FakeTokenizer())
    assert probe.available(_ctx({"train": train}))
    assert not probe.available(_ctx({"train": train}, model=""))
    assert not probe.available(_ctx({}))
    assert not probe.available(_ctx({"train": str(tmp_path / "missing.jsonl")}))
    assert probe.estimate_cost(_ctx({"train": train})) == 0.0


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("", "no examples"),
        ("\n\n", "no examples"),
        ("not json\n", "not JSON"),
        ('{"text": "x"}\n', "no messages"),
        ('{"messages": []}\n', "no messages"),
        ('{"messages": [{"role": "user"}]}\n', "role and content"),
        ("[1, 2]\n", "no messages"),
    ],
)
def test_malformed_files_are_errors(tmp_path, content, error):
    path = tmp_path / "train.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        read_chat_jsonl(path)


def test_signatures_cover_the_probe_outcomes():
    assert "example_length" in load_signatures()
    assert set(outcomes("example_length")) == set(ExampleLength.outcomes)
    assert likelihood("example_length", "truncation", "truncated") == 0.90
    assert likelihood("example_length", "lr_too_low", "truncated") == 0.05
