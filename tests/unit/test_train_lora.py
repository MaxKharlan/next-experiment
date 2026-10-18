import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from jobs.kaggle.train_lora import PAD_MULTIPLE, collate, encode, lr_factor


class _Tok:
    """Chat template that joins role:content lines; one token per character."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, **_):
        text = "".join(f"{m['role']}:{m['content']}\n" for m in messages)
        return text + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) for c in text]}


CHAT = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "ok"},
]


def test_encode_masks_the_prompt():
    row = encode(_Tok(), CHAT, context_length=100)
    answer = [i for i, label in zip(row["input_ids"], row["labels"]) if label != -100]
    assert "".join(map(chr, answer)) == "ok\n"


def test_encode_truncates_to_context_length():
    row = encode(_Tok(), CHAT, context_length=5)
    assert len(row["input_ids"]) == len(row["labels"]) == 5


def test_collate_pads_to_multiple():
    rows = [encode(_Tok(), CHAT, 100), {"input_ids": [1, 2], "labels": [-100, 2]}]
    batch = collate(rows, pad_id=0, device="cpu")
    assert batch["input_ids"].shape[1] % PAD_MULTIPLE == 0
    assert batch["labels"][1, 2:].eq(-100).all()
    assert batch["attention_mask"][1].sum() == 2


def test_lr_factor():
    assert lr_factor(0, total=10, warmup=0, schedule="linear") == 1.0
    assert lr_factor(5, total=10, warmup=0, schedule="linear") == 0.5
    assert lr_factor(0, total=10, warmup=2, schedule="constant") == 0.5
    assert lr_factor(9, total=10, warmup=2, schedule="constant") == 1.0


class _TinyLM(torch.nn.Module):
    """Deterministic CPU next-token model; no downloads or dropout."""

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 8, bias=False)

    def forward(self, input_ids, attention_mask):
        from types import SimpleNamespace

        from torch.nn import functional

        inputs = functional.one_hot(input_ids % 8, num_classes=8).float()
        return SimpleNamespace(logits=self.projection(inputs))


def _gradients(rows, *, faulty=False):
    from jobs.kaggle.train_lora import backward_microbatches

    torch.manual_seed(42)
    model = _TinyLM()
    batches = [collate(group, 0, "cpu") for group in rows]
    loss, correct, tokens = backward_microbatches(
        model, batches, microbatch_mean=faulty
    )
    return model.projection.weight.grad.clone(), loss, correct, tokens


@pytest.mark.parametrize("sizes", [(1, 5, 2), (3, 3, 3), (1, 4), (2,)])
def test_accumulation_matches_full_batch(sizes):
    rows = [
        {"input_ids": [0] + [i + 1] * n, "labels": [-100] + [i + 1] * n}
        for i, n in enumerate(sizes)
    ]
    full = _gradients([rows])
    accumulated = _gradients([[row] for row in rows])
    torch.testing.assert_close(full[0], accumulated[0])
    assert full[1] == pytest.approx(accumulated[1])
    assert full[2:] == accumulated[2:]
    faulty = _gradients([[row] for row in rows], faulty=True)
    if len(set(sizes)) == 1:
        torch.testing.assert_close(full[0], faulty[0])
    else:
        assert not torch.allclose(full[0], faulty[0], atol=1e-5)


def test_accumulation_rejects_empty_targets_before_backward():
    from jobs.kaggle.train_lora import backward_microbatches

    model = _TinyLM()
    empty = collate([{"input_ids": [1, 2], "labels": [-100, -100]}], 0, "cpu")
    valid = collate([{"input_ids": [1, 2], "labels": [-100, 2]}], 0, "cpu")
    for batches in ([], [valid, empty]):
        with pytest.raises(ValueError, match="supervised"):
            backward_microbatches(model, batches)
        assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize(
    "code_fault", [None, "eos_as_pad", "prompt_loss", "double_shift"]
)
def test_training_accumulation_steps_and_remainder(tmp_path, monkeypatch, code_fault):
    import json

    from jobs.kaggle import train_lora as trainer

    class Tokenizer(_Tok):
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, text, add_special_tokens):
            return {"input_ids": [ord(c) % 8 for c in text]}

    class Model(_TinyLM):
        def save_pretrained(self, path):
            pass

    original_evaluate = trainer.evaluate

    def checked_evaluate(model, rows, *args):
        assert all(row["labels"][-1] == 2 for row in rows)
        return original_evaluate(model, rows, *args)

    monkeypatch.setattr(trainer, "evaluate", checked_evaluate)
    models = []

    def load(*args, **kwargs):
        model = Model()
        models.append(model)
        return model

    monkeypatch.setattr(
        trainer.AutoTokenizer, "from_pretrained", lambda *_: Tokenizer()
    )
    monkeypatch.setattr(trainer.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(trainer, "get_peft_model", lambda model, _: model)
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(
                {"messages": [CHAT[0], {"role": "assistant", "content": "a" * n}]}
            )
            for n in [1, 5, 2, 7, 3]
        )
    )
    summaries = []
    for bs, accumulation in [(3, 1), (1, 3)]:
        summaries.append(
            trainer.train(
                data,
                data,
                tmp_path / str(bs),
                device="cpu",
                seed=42,
                code_fault=code_fault,
                hp={"batch_size": bs, "gradient_accumulation_steps": accumulation},
            )
        )
    assert [s["steps"] for s in summaries] == [2, 2]
    assert summaries[0]["answer_tokens"] == summaries[1]["answer_tokens"]
    for full, accumulated in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(full, accumulated)
    assert summaries[1]["gradient_accumulation_steps"] == 3
    assert summaries[1]["code_fault"] == code_fault
    assert summaries[1]["answer_tokens"] == (18 if code_fault == "eos_as_pad" else 23)
    audit = summaries[1]["label_audit"]
    if code_fault == "double_shift":
        assert audit is None
        assert summaries[1]["label_shift_audit"]["status"] == "double_shift"
        assert summaries[1]["label_shift_audit"]["preshift_mismatches"] == 0
    else:
        assert audit["reference_eos_targets"] == 5
        assert audit["removed_eos_targets"] == (5 if code_fault == "eos_as_pad" else 0)
    if code_fault == "prompt_loss":
        prompt = summaries[1]["prompt_label_audit"]
        assert prompt["supervised_prompt_tokens"] == 85
        assert prompt["answer_targets"] == 23
        assert summaries[1]["supervised_tokens"] == 108
    else:
        assert summaries[1]["supervised_tokens"] == summaries[1]["answer_tokens"]


@pytest.mark.parametrize("accumulation", [0, -1, True, 1.5])
def test_invalid_accumulation_fails_before_loading(accumulation):
    from jobs.kaggle.train_lora import train

    with pytest.raises(ValueError, match="positive integer"):
        train(
            "missing",
            "missing",
            "unused",
            hp={"gradient_accumulation_steps": accumulation},
        )


@pytest.mark.parametrize("fault", ["unknown", "microbatch_mean"])
def test_invalid_code_fault_fails_before_loading(fault):
    from jobs.kaggle.train_lora import train

    with pytest.raises(ValueError):
        train("missing", "missing", "unused", code_fault=fault)


def test_collate_with_eos_padding_keeps_real_eos_targets():
    rows = [
        {"input_ids": [4, 5, 2], "labels": [-100, 5, 2]},
        {"input_ids": [4, 2], "labels": [-100, 2]},
    ]
    batch = collate(rows, pad_id=2, device="cpu")
    assert batch["labels"][0, 2] == 2
    assert batch["labels"][1, 1] == 2
    assert batch["labels"][batch["attention_mask"] == 0].eq(-100).all()


@pytest.mark.parametrize(
    "eos_id,labels,message",
    [
        (None, [-100, 2], "EOS token"),
        (2, [-100, 3], "removed no"),
        (2, [-100, 2], "no supervised"),
    ],
)
def test_eos_fault_rejects_unusable_data_before_model_loading(
    tmp_path, monkeypatch, eos_id, labels, message
):
    from types import SimpleNamespace

    from jobs.kaggle import train_lora as trainer

    monkeypatch.setattr(
        trainer.AutoTokenizer,
        "from_pretrained",
        lambda *_: SimpleNamespace(pad_token_id=0, eos_token_id=eos_id),
    )
    monkeypatch.setattr(
        trainer,
        "encode",
        lambda *_: {"input_ids": [1, labels[-1]], "labels": labels.copy()},
    )
    monkeypatch.setattr(
        trainer.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **kw: pytest.fail("Model must not load for invalid fault data"),
    )
    path = tmp_path / "data.jsonl"
    path.write_text('{"messages": []}\n')
    with pytest.raises(ValueError, match=message):
        trainer.train(
            path, path, tmp_path / "out", code_fault="eos_as_pad", device="cpu"
        )


def test_prompt_loss_does_not_supervise_padding():
    from bench.faults.labels import include_prompt_labels

    rows = [{"input_ids": [3, 4, 5], "labels": [-100, -100, 5]}]
    batch = collate(include_prompt_labels(rows), pad_id=0, device="cpu")
    assert batch["labels"][0, :3].tolist() == [3, 4, 5]
    assert batch["labels"][batch["attention_mask"] == 0].eq(-100).all()


def test_prompt_fault_changes_gradient():
    from bench.faults.labels import include_prompt_labels

    rows = [{"input_ids": [1, 2, 3, 4], "labels": [-100, -100, -100, 4]}]
    healthy = _gradients([rows])
    faulty = _gradients([include_prompt_labels(rows)])
    assert healthy[3] == 1
    assert faulty[3] == 3
    assert not torch.allclose(healthy[0], faulty[0])


def test_prompt_fault_without_prompt_targets_fails_before_model_loading(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from jobs.kaggle import train_lora as trainer

    monkeypatch.setattr(
        trainer.AutoTokenizer,
        "from_pretrained",
        lambda *_: SimpleNamespace(pad_token_id=0, eos_token_id=2),
    )
    monkeypatch.setattr(
        trainer, "encode", lambda *_: {"input_ids": [1, 2], "labels": [-100, 2]}
    )
    monkeypatch.setattr(
        trainer.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **kw: pytest.fail("Model must not load"),
    )
    path = tmp_path / "data.jsonl"
    path.write_text('{"messages": []}\n')
    with pytest.raises(ValueError, match="added no"):
        trainer.train(
            path, path, tmp_path / "out", code_fault="prompt_loss", device="cpu"
        )


def test_double_shift_uses_token_two_positions_ahead():
    from types import SimpleNamespace

    from bench.faults.labels import shift_labels_once
    from jobs.kaggle.train_lora import loss_and_accuracy

    class NextTokenModel:
        def __call__(self, input_ids, attention_mask):
            logits = torch.full((*input_ids.shape, 8), -10.0)
            logits.scatter_(-1, ((input_ids + 1) % 8).unsqueeze(-1), 10.0)
            return SimpleNamespace(logits=logits)

    rows = [{"input_ids": [1, 2, 3, 4], "labels": [-100, 2, 3, 4]}]
    healthy = loss_and_accuracy(NextTokenModel(), collate(rows, 0, "cpu"))
    faulty_batch = collate(shift_labels_once(rows), 0, "cpu")
    assert faulty_batch["labels"][0, 1:4].tolist() == [3, 4, -100]
    faulty = loss_and_accuracy(NextTokenModel(), faulty_batch)
    assert healthy[0].item() < 1e-5
    assert healthy[1:] == (3, 3)
    assert faulty[0].item() > 19
    assert faulty[1:] == (0, 2)
    assert faulty_batch["labels"][faulty_batch["attention_mask"] == 0].eq(-100).all()


@pytest.mark.parametrize(
    "labels,message", [([-100], "did not change"), ([-100, 2], "no supervised")]
)
def test_double_shift_rejects_unusable_rows_before_loading(
    tmp_path, monkeypatch, labels, message
):
    from types import SimpleNamespace

    from jobs.kaggle import train_lora as trainer

    monkeypatch.setattr(
        trainer.AutoTokenizer,
        "from_pretrained",
        lambda *_: SimpleNamespace(pad_token_id=0, eos_token_id=2),
    )
    monkeypatch.setattr(
        trainer,
        "encode",
        lambda *_: {
            "input_ids": list(range(1, len(labels) + 1)),
            "labels": labels.copy(),
        },
    )
    monkeypatch.setattr(
        trainer.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **kw: pytest.fail("Model must not load"),
    )
    path = tmp_path / "data.jsonl"
    path.write_text('{"messages": []}\n')
    with pytest.raises(ValueError, match=message):
        trainer.train(
            path, path, tmp_path / "out", code_fault="double_shift", device="cpu"
        )
