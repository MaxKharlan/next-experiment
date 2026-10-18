import json

import pytest

from core.case import EvaluatorSpec
from core.eval import breakdown, compare_checkpoints, evaluate, max_new_tokens
from core.eval import generate as gen
from core.eval.breakdown import by_label, by_prompt_length, by_source, stopping
from core.eval.data import read_examples
from core.eval.metrics import exact_match, get_metric
from core.eval.records import EvalResult, ExampleResult

EOS = 99
SPEC = EvaluatorSpec(name="intent_label", metric="exact_match")


class FakeEncoding(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    """Words are tokens: each word gets an id, EOS is 99 and decodes to nothing."""

    eos_token_id = EOS

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.templates = []

    def ids(self, text):
        return [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append(kwargs)
        return " ".join(m["content"] for m in messages) + " <assistant>"

    def __call__(self, prompt, add_special_tokens, return_tensors):
        assert add_special_tokens is False
        return FakeEncoding(input_ids=[self.ids(prompt)])

    def decode(self, ids, skip_special_tokens):
        words = {i: w for w, i in self.vocab.items()}
        return " ".join(words[i] for i in ids if not (skip_special_tokens and i == EOS))


class FakeModel:
    """Answers with the next scripted reply; None in the script means no EOS."""

    def __init__(self, tok, replies, eos=True):
        self.tok, self.replies, self.eos = tok, list(replies), eos
        self.calls = []
        self.evaluated = False

    def eval(self):
        self.evaluated = True

    def to(self, device):
        return self

    def generate(self, input_ids, max_new_tokens, do_sample):
        self.calls.append((max_new_tokens, do_sample))
        new = self.tok.ids(self.replies.pop(0))[:max_new_tokens]
        if self.eos and len(new) < max_new_tokens:
            new.append(EOS)
        return [input_ids[0] + new]


def _write(path, rows):
    lines = []
    for query, label, *source in rows:
        record = {
            "messages": [
                {"role": "system", "content": "label it"},
                {"role": "user", "content": query},
                {"role": "assistant", "content": label},
            ]
        }
        if source:
            record["source"] = source[0]
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _result(
    line, prompt_tokens, correct=True, expected="a", eos=True, tokens=1, source=None
):
    return ExampleResult(
        line=line,
        expected=expected,
        answer=expected if correct else "x",
        score=1.0 if correct else 0.0,
        prompt_tokens=prompt_tokens,
        answer_tokens=tokens,
        stopped_on_eos=eos,
        hit_limit=not eos,
        source=source,
    )


# data and metrics


def test_examples_keep_line_numbers_prompt_reference_and_source(tmp_path):
    path = tmp_path / "check.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": " card "},
                ],
                "metadata": {"source": "app"},
            }
        )
        + "\n\n"
        + json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "r"},
                    {"role": "assistant", "content": "pin"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    first, second = read_examples(path)
    assert (first.line, first.expected, first.source) == (1, " card ", "app")
    assert first.prompt == [{"role": "user", "content": "q"}]
    assert (second.line, second.source) == (3, None)


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("", "no examples"),
        ("nope\n", "not JSON"),
        (
            '{"messages": [{"role": "assistant", "content": "a"}]}\n',
            "prompt and an answer",
        ),
        (
            '{"messages": [{"role": "user", "content": "q"}, {"role": "user", "content": "a"}]}\n',
            "not assistant",
        ),
        (
            '{"messages": [{"role": "user"}, {"role": "assistant", "content": "a"}]}\n',
            "role and content",
        ),
    ],
)
def test_malformed_files_are_errors(tmp_path, content, error):
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        read_examples(path)


def test_exact_match_strips_like_the_old_evaluator():
    assert exact_match(" card_arrival\n", "card_arrival") == 1.0
    assert exact_match("card arrival", "card_arrival") == 0.0
    assert get_metric("exact_match") is exact_match
    with pytest.raises(ValueError, match="unknown metric 'json_fields'"):
        get_metric("json_fields")


@pytest.mark.parametrize("value", [0, -3, 32.0, "32", True])
def test_max_new_tokens_must_be_a_positive_integer(value):
    assert max_new_tokens(SPEC) == 32
    spec = EvaluatorSpec(
        name="t", metric="exact_match", parameters={"max_new_tokens": 8}
    )
    assert max_new_tokens(spec) == 8
    bad = EvaluatorSpec(
        name="t", metric="exact_match", parameters={"max_new_tokens": value}
    )
    with pytest.raises(ValueError, match="max_new_tokens"):
        max_new_tokens(bad)


# generation


def test_answers_are_greedy_scored_and_record_how_they_end(tmp_path):
    examples = read_examples(
        _write(
            tmp_path / "c.jsonl",
            [("lost my card", "lost_card"), ("pin please", "pin"), ("hi", "greet")],
        )
    )
    tok = FakeTokenizer()
    model = FakeModel(tok, ["lost_card", "wrong", "greet greet greet"])
    results = gen.answers(model, tok, examples, "cpu", exact_match, max_new_tokens=3)

    assert model.evaluated
    assert model.calls == [(3, False)] * 3
    assert tok.templates[0] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    first, second, third = results
    assert (first.answer, first.score, first.stopped_on_eos, first.answer_tokens) == (
        "lost_card",
        1.0,
        True,
        1,
    )
    assert first.prompt_tokens == 6  # label it lost my card <assistant>
    assert (second.score, second.line) == (0.0, 2)
    # three tokens fill the limit, so there is no room for EOS
    assert (third.stopped_on_eos, third.hit_limit, third.answer_tokens) == (
        False,
        True,
        3,
    )


def test_an_answer_without_eos_below_the_limit_is_not_cut(tmp_path):
    tok = FakeTokenizer()
    model = FakeModel(tok, ["pin"], eos=False)
    example = read_examples(_write(tmp_path / "c.jsonl", [("pin please", "pin")]))[0]
    result = gen.answer(model, tok, example, "cpu", exact_match, max_new_tokens=8)
    assert (result.stopped_on_eos, result.hit_limit, result.score) == (
        False,
        False,
        1.0,
    )


def test_eos_ids_join_the_generation_config_and_the_tokenizer():
    class Config:
        eos_token_id = (7, 8)

    class Model:
        generation_config = Config()

    assert gen.eos_ids(Model(), FakeTokenizer()) == {7, 8, EOS}
    assert gen.eos_ids(object(), FakeTokenizer()) == {EOS}


def test_peft_names_add_the_prefix_once():
    names = gen.peft_names(
        {"model.q.lora_A.weight": 1, "base_model.model.k.lora_B.weight": 2}
    )
    assert sorted(names) == [
        "base_model.model.k.lora_B.weight",
        "base_model.model.model.q.lora_A.weight",
    ]


def test_load_model_needs_an_adapter_or_a_base_model():
    with pytest.raises(ValueError, match="adapter directory or a base model"):
        gen.load_model()


def test_evaluate_loads_once_and_returns_every_record(tmp_path, monkeypatch):
    data = _write(
        tmp_path / "c.jsonl", [("lost my card", "lost_card"), ("pin please", "pin")]
    )
    tok = FakeTokenizer()
    model = FakeModel(tok, ["lost_card", "nope"])
    loads = []

    def load_model(adapter, base_model, template_dir):
        loads.append((adapter, base_model, template_dir))
        return model, tok, "Qwen/Qwen3-0.6B"

    monkeypatch.setattr(gen, "load_model", load_model)
    monkeypatch.setattr(gen, "pick_device", lambda device: "cpu")
    result = evaluate(data, SPEC, adapter=tmp_path / "adapter")

    assert loads == [(tmp_path / "adapter", None, None)]
    assert (result.value, result.metric, result.max_new_tokens) == (
        0.5,
        "exact_match",
        32,
    )
    assert (result.model, result.adapter) == (
        "Qwen/Qwen3-0.6B",
        str(tmp_path / "adapter"),
    )
    assert [e.line for e in result.wrong()] == [2]
    assert (
        json.loads(json.dumps(result.to_dict()))["examples"][0]["answer"] == "lost_card"
    )


def test_evaluate_checks_the_metric_before_loading_a_model(tmp_path, monkeypatch):
    data = _write(tmp_path / "c.jsonl", [("q", "a")])
    monkeypatch.setattr(gen, "load_model", lambda *a: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="unknown metric"):
        evaluate(data, EvaluatorSpec(name="t", metric="json_fields"), base_model="m")


# breakdown and checkpoints


def test_errors_by_prompt_length_quartiles():
    examples = [
        _result(i, tokens, correct=tokens < 70)
        for i, tokens in enumerate([10, 20, 30, 40, 50, 60, 70, 80], 1)
    ]
    rows = by_prompt_length(examples)
    assert [r["tokens"] for r in rows] == [[10, 20], [21, 40], [41, 60], [61, 80]]
    assert [r["examples"] for r in rows] == [2, 2, 2, 2]
    assert [r["errors"] for r in rows] == [0, 0, 0, 2]
    # equal lengths share one bucket
    same = by_prompt_length([_result(i, 5) for i in range(4)])
    assert same == [{"tokens": [5, 5], "examples": 4, "errors": 0, "error_rate": 0.0}]


def test_errors_by_label_and_source():
    examples = [
        _result(1, 10, correct=False, expected="pin", source="app"),
        _result(2, 10, correct=False, expected="pin", source="web"),
        _result(3, 10, expected="card", source="app"),
    ]
    assert by_label(examples)[0] == {
        "value": "pin",
        "examples": 2,
        "errors": 2,
        "error_rate": 1.0,
    }
    assert [r["value"] for r in by_source(examples)] == ["app", "web"]
    assert by_source([_result(1, 10)]) == []


def test_stopping_shares():
    ends = stopping([_result(1, 5, tokens=2), _result(2, 5, eos=False, tokens=32)])
    assert ends == {
        "examples": 2,
        "no_eos_share": 0.5,
        "hit_limit_share": 0.5,
        "answer_tokens_mean": 17.0,
        "answer_tokens_max": 32,
    }


def test_breakdown_of_a_result():
    examples = [_result(1, 5), _result(2, 9, correct=False)]
    result = EvalResult.from_examples(
        "exact_match", examples, model="m", data="d", max_new_tokens=32
    )
    report = breakdown(result)
    assert (report["value"], report["errors"]) == (0.5, 1)
    assert set(report) >= {"by_prompt_length", "by_label", "by_source", "stopping"}
    with pytest.raises(ValueError):
        EvalResult.from_examples(
            "exact_match", [], model="m", data="d", max_new_tokens=32
        )


def test_checkpoints_are_ordered_by_step_and_the_best_epoch_is_named(tmp_path):
    values = {"s10": 0.4, "s20": 0.6, "s30": 0.5}

    def score(path):
        value = values[path.name]
        examples = [_result(1, 5, correct=True)] * int(value * 10) + [
            _result(2, 5, correct=False)
        ] * (10 - int(value * 10))
        return EvalResult.from_examples(
            "exact_match", examples, model="m", data="d", max_new_tokens=32
        )

    table = compare_checkpoints(
        [(30, tmp_path / "s30"), (10, tmp_path / "s10"), (20, tmp_path / "s20")], score
    )
    assert [(r["step"], r["epoch"]) for r in table["rows"]] == [
        (10, 1),
        (20, 2),
        (30, 3),
    ]
    assert [r["value"] for r in table["rows"]] == pytest.approx([0.4, 0.6, 0.5])
    assert (table["best_step"], table["best_epoch"], table["last_is_best"]) == (
        20,
        2,
        False,
    )
    lower = compare_checkpoints(
        [(10, tmp_path / "s10"), (20, tmp_path / "s20")], score, higher_is_better=False
    )
    assert lower["best_step"] == 10
    with pytest.raises(ValueError):
        compare_checkpoints([], score)
