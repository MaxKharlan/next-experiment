import pytest

torch = pytest.importorskip("torch")

from core.training.parameters import audit_trainable_parameters


def test_inventory_and_exact_lora_path_components():
    model = torch.nn.Module()
    model.lora_A = torch.nn.Linear(3, 2, bias=False)
    model.lora_B = torch.nn.Linear(2, 4, bias=False)
    model.base = torch.nn.Linear(3, 4, bias=False)
    model.base.requires_grad_(False)
    model.not_lora_A = torch.nn.Linear(1, 1, bias=False)
    report = audit_trainable_parameters(model)
    assert report["total_parameters"] == 27
    assert report["trainable_parameters"] == 15
    assert report["frozen_parameters"] == 12
    assert report["trainable_lora_parameters"] == 14
    assert report["trainable_non_lora_parameters"] == 1
    assert report["findings"] == ["trainable_non_lora_parameters"]
    assert sum(t["numel"] for t in report["trainable_tensors"]) == 15
    assert model.base.weight.requires_grad is False


def test_tied_weights_are_counted_once_and_empty_model_is_reported():
    model = torch.nn.Module()
    model.first = torch.nn.Linear(2, 3, bias=False)
    model.second = model.first
    assert audit_trainable_parameters(model)["total_parameters"] == 6
    empty = audit_trainable_parameters(torch.nn.Module())
    assert empty["trainable_fraction"] is None
    assert "no_trainable_parameters" in empty["findings"]


def test_real_peft_adapter_frozen_base_and_deliberate_misconfiguration():
    peft = pytest.importorskip("peft")
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=16,
            vocab_size=16,
        )
    )
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["c_attn"],
            bias="none",
            fan_in_fan_out=True,
            task_type="CAUSAL_LM",
        ),
    )
    report = audit_trainable_parameters(model)
    assert report["findings"] == []
    assert report["trainable_lora_parameters"] == 64
    assert report["frozen_parameters"] > 0
    model.base_model.model.transformer.wte.weight.requires_grad_(True)
    assert audit_trainable_parameters(model)["trainable_non_lora_parameters"] == 128
    model.requires_grad_(False)
    frozen = audit_trainable_parameters(model)
    assert frozen["lora_parameters"] == 64
    assert frozen["trainable_parameters"] == 0
    assert frozen["findings"] == [
        "no_trainable_parameters",
        "no_trainable_lora_parameters",
    ]
