import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

from jobs.eval_local.exact_match import peft_names


def test_token_factory_names_get_the_peft_prefix():
    tf = "model.layers.0.mlp.down_proj.lora_A.weight"
    peft = "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"
    assert set(peft_names({tf: 1, peft: 2})) == {f"base_model.model.{tf}", peft}
