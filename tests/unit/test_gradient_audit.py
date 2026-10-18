import pytest

torch = pytest.importorskip("torch")

from core.training.gradients import audit_gradients


def test_distinguishes_missing_zero_nonfinite_and_nonzero_without_mutation():
    model = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.ones(2)) for _ in range(5)]
    )
    model[0].grad = torch.zeros(2)
    model[1].grad = torch.tensor([1.0, 0.0])
    model[2].grad = torch.tensor([float("inf"), 0.0])
    model[4].requires_grad_(False)
    original = model[1].grad.clone()
    report = audit_gradients(model)
    assert report["tensor_counts"] == {
        "missing": 1,
        "zero": 1,
        "nonzero": 1,
        "nonfinite": 1,
    }
    assert report["findings"] == ["some_gradients_missing", "nonfinite_gradients"]
    torch.testing.assert_close(model[1].grad, original)
    assert model[3].grad is None


def test_empty_missing_and_zero_are_distinct():
    assert audit_gradients(torch.nn.Module())["findings"] == ["no_trainable_tensors"]
    model = torch.nn.Linear(2, 2, bias=False)
    assert audit_gradients(model)["findings"] == ["all_gradients_missing"]
    model.weight.grad = torch.zeros_like(model.weight)
    assert audit_gradients(model)["findings"] == ["all_gradients_zero"]


def test_sparse_embedding_gradient():
    model = torch.nn.Embedding(4, 2, sparse=True)
    model(torch.tensor([1, 1])).sum().backward()
    assert audit_gradients(model)["tensor_counts"]["nonzero"] == 1


def test_real_peft_first_backward_has_zero_a_and_nonzero_b():
    peft = pytest.importorskip("peft")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(42)
    model = peft.get_peft_model(
        GPT2LMHeadModel(
            GPT2Config(n_layer=1, n_head=1, n_embd=8, n_positions=16, vocab_size=16)
        ),
        peft.LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["c_attn"],
            bias="none",
            fan_in_fan_out=True,
            task_type="CAUSAL_LM",
        ),
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=ids, labels=ids).loss.backward()
    report = audit_gradients(model)
    assert report["findings"] == []
    assert report["tensor_counts"] == {
        "missing": 0,
        "zero": 1,
        "nonzero": 1,
        "nonfinite": 0,
    }
    assert (
        next(t for t in report["tensors"] if ".lora_A." in t["name"])["status"]
        == "zero"
    )
