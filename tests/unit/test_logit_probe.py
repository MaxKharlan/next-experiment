import math
from types import SimpleNamespace

import pytest

from core.eval.data import Example
from core.probes.level1.logits import (
    TokenRegions,
    aggregate,
    encode_regions,
    inspect_logits,
    loss_summary,
)

torch = pytest.importorskip("torch")


def regions():
    return TokenRegions(
        input_ids=(0, 1, 2, 3),
        prompt=(1,),
        answer=(2,),
        template_special=(3,),
        answer_end=3,
        eos_ids=(3,),
    )


def test_uniform_logits_have_known_nll_and_eos_probability():
    result = inspect_logits(torch.zeros(4, 4), regions())
    for name in ("prompt", "answer", "template_special"):
        assert result[name]["mean_nll"] == pytest.approx(math.log(4))
        assert result[name]["perplexity"] == pytest.approx(4)
    assert result["eos_probability_after_reference"] == pytest.approx(0.25)


def test_causal_shift_uses_preceding_position_not_same_position():
    logits = torch.full((4, 4), -10.0)
    logits[0, 1] = logits[1, 2] = logits[2, 3] = 10
    logits[3, 0] = 10
    result = inspect_logits(logits, regions())
    assert result["answer"]["mean_nll"] < 1e-5
    assert result["eos_probability_after_reference"] > 0.999


def test_overlap_nonfinite_and_wrong_eos_boundary_rejected():
    original = regions()
    overlapping = TokenRegions(original.input_ids, (1, 2), (2,), (3,), 3, (3,))
    with pytest.raises(ValueError, match="overlap"):
        inspect_logits(torch.zeros(4, 4), overlapping)
    with pytest.raises(ValueError, match="Nonfinite"):
        inspect_logits(torch.full((4, 4), float("nan")), original)
    wrong = TokenRegions(original.input_ids, (1,), (2,), (3,), 3, (2,))
    with pytest.raises(ValueError, match="boundary"):
        inspect_logits(torch.zeros(4, 4), wrong)


def test_aggregate_weights_tokens_and_keeps_empty_groups_explicit():
    one = inspect_logits(torch.zeros(4, 4), regions())
    two = {**one, "answer": loss_summary(12, 3), "prompt": loss_summary(0, 0)}
    result = aggregate([one, two])
    assert result["answer"]["mean_nll"] == pytest.approx((math.log(4) + 12) / 4)
    assert result["prompt"]["tokens"] == 1
    assert loss_summary(0, 0)["perplexity"] is None
    assert loss_summary(1000, 1)["perplexity_overflow"] is True


def test_token_boundaries_reject_truncation_and_rendering_mismatch():
    tokens = {"prompt": [0, 1], "full": [0, 1, 2, 3, 4]}
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kw: (
            "prompt" if kw["add_generation_prompt"] else "full"
        ),
        encode=lambda text, **kw: tokens[text],
        eos_token_id=3,
        all_special_ids=[0, 3],
    )
    example = Example(1, [{"role": "user", "content": "question"}], "answer")
    encoded = encode_regions(tokenizer, example, context_length=5)
    assert encoded.prompt == (1,)
    assert encoded.answer == (2,)
    assert encoded.template_special == (3,)
    with pytest.raises(ValueError, match="overflow"):
        encode_regions(tokenizer, example, context_length=4)
    tokens["prompt"] = [9, 1]
    with pytest.raises(ValueError, match="prefix"):
        encode_regions(tokenizer, example, context_length=5)
