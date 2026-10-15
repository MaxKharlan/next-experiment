import pytest

pytest.importorskip("torch")
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
