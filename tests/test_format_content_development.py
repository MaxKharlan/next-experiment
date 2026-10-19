import json
import math
import random
from types import SimpleNamespace

import pytest

from bench.tasks.format_content import alternatives, build, direct, make_row
from core.eval.format_content import (
    completion_logprob,
    generation_metrics,
    selected_score,
)
from core.eval.structured import validate_answer
from scripts.format_content_development import gate_failures


class CharTokenizer:
    def encode(self, text, **_):
        # Mimic single-token enum values to exercise four equal-token choices.
        import re

        return re.findall(r'"[a-z_]+"|.', text)


def test_shared_anchors_and_matched_corruption(tmp_path):
    data = build(tmp_path / "data", CharTokenizer())

    def rows(arm, role):
        return [json.loads(s) for s in open(data[f"{arm}_{role}"]["path"])]

    for role, n in (("train", 72), ("validation", 12)):
        healthy, cut, wrong, fmt = [
            rows(a, role) for a in ("healthy", "cut", "wrong", "format")
        ]
        assert healthy[:n] == cut[:n] == wrong[:n] == fmt[:n]
        changed = []
        for i, (h, c, w) in enumerate(zip(healthy, cut, wrong, strict=True)):
            assert h["messages"][:-1] == c["messages"][:-1] == w["messages"][:-1]
            original = h["messages"][-1]["content"]
            bad = w["messages"][-1]["content"]
            validate_answer("json_extraction", bad)
            if c != h:
                changed.append(i)
                assert c["messages"][-1]["content"] == original[: len(original) // 2]
                assert bad != original
                assert i >= n
            else:
                assert w == h
        assert len(changed) == n
        for row in fmt:
            fields = dict(
                line.split("=", 1)
                for line in row["messages"][1]["content"].splitlines()[1:]
            )
            assert {k: json.loads(v) for k, v in fields.items()} == json.loads(
                row["messages"][-1]["content"]
            )


def test_choice_candidates_and_wrapper_separation():
    row = make_row("train", 0, random.Random(42))
    expected = row["messages"][-1]["content"]
    choice = alternatives(expected, CharTokenizer(), random.Random(1))
    assert len(set(choice["choices"])) == 4
    assert choice["choices"][choice["correct_index"]] == expected
    for c in choice["choices"]:
        value = json.loads(c)
        for k in ("ticket_id", "due_date", "callback"):
            assert value[k] == json.loads(expected)[k]
    assert generation_metrics("```json\n" + expected + "\n```", expected) == {
        "strict_exact": 0,
        "schema_valid": 0,
        "unwrapped_exact": 1,
    }
    wrong = json.loads(expected)
    wrong["product"] = "scanner" if wrong["product"] != "scanner" else "router"
    assert generation_metrics(json.dumps(wrong), expected) == {
        "strict_exact": 0,
        "schema_valid": 1,
        "unwrapped_exact": 0,
    }
    assert selected_score([1, 1, 0, 0], 0) == 0
    with pytest.raises(ValueError):
        selected_score([math.nan, 0], 0)
    assert direct(row)["messages"][-1]["content"] == expected


def test_conditional_probability_alignment_excludes_prompt():
    torch = pytest.importorskip("torch")

    class Model:
        def __call__(self, input_ids, use_cache):
            assert input_ids.tolist() == [[0, 1, 2, 3]]
            # Prompt position 0 deliberately terrible for the prompt token 1.
            logits = torch.tensor(
                [
                    [
                        [0.0, -99.0, 0.0, 0.0],
                        [0.0, 0.0, 3.0, 0.0],
                        [0.0, 0.0, 0.0, 2.0],
                        [99.0, 0.0, 0.0, 0.0],
                    ]
                ]
            )
            return SimpleNamespace(logits=logits)

    expected = 3 - math.log(math.exp(3) + 3) + 2 - math.log(math.exp(2) + 3)
    assert completion_logprob(Model(), [0, 1], [2, 3], "cpu") == pytest.approx(
        expected, abs=1e-6
    )


def test_gate_detects_format_content_and_cut_separately():
    groups = {
        a: {
            "format": {"strict_exact": [1] * 3},
            "content": {"choice_exact": [v] * 3, "strict_exact": [s] * 3},
        }
        for a, v, s in [
            ("format", 0.4, 0.3),
            ("healthy", 0.8, 0.7),
            ("cut", 0.6, 0.5),
            ("wrong", 0.5, 0.4),
        ]
    }
    rule = {
        "format_exact_minimum": 0.9,
        "healthy_choice_minimum": 0.35,
        "healthy_choice_maximum": 0.95,
        "healthy_minus_format_choice": 0.1,
        "healthy_minus_wrong_choice": 0.1,
        "healthy_minus_cut_strict": 0.08,
    }
    assert gate_failures(groups, rule) == []
    groups["healthy"]["content"]["choice_exact"] = [0.5] * 3
    assert "insufficient_choice_gap_wrong" in gate_failures(groups, rule)
    groups["cut"]["content"]["strict_exact"] = [0.8] * 3
    assert "insufficient_cut_sensitivity" in gate_failures(groups, rule)
