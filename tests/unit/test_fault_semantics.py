"""Keep upstream answer damage separate from trainer truncation."""

from core.agent.catalog import allowed_classes, catalog_text
from core.agent.schemas import Hypothesis
from core.agent.steps import clean_hypotheses, load_prompt


def hypothesis(fault_class):
    return Hypothesis(
        id="candidate",
        fault_class=fault_class,
        location="train.jsonl",
        prior=0.4,
        rationale="Observed incomplete assistant answers.",
    )


def test_token_factory_accepts_answer_damage_not_trainer_truncation():
    found = clean_hypotheses(
        [hypothesis("answer_truncation"), hypothesis("truncation")],
        "A",
    )
    assert [item.fault_class for item in found] == ["answer_truncation"]
    assert found[0].prior == 0.4
    assert "truncation" not in allowed_classes("A")


def test_custom_training_keeps_both_distinct_causes():
    assert {"truncation", "answer_truncation"} <= allowed_classes("B")


def test_legacy_hypotheses_remain_readable_without_relabeling():
    saved = hypothesis("truncation").model_dump_json()
    restored = Hypothesis.model_validate_json(saved)
    assert restored.fault_class == "truncation"


def test_current_prompt_distinguishes_evidence_from_confirmation():
    prompt = load_prompt("hypotheses", "v4")
    rendered = prompt.render(catalog=catalog_text("A"))
    assert prompt.id == "hypotheses-v4"
    assert "answer_truncation" in rendered
    assert "fails dataset processing" in rendered
    assert "not causal proof" in rendered
    assert "- truncation (" not in rendered


def test_legacy_prompt_version_remains_available():
    assert load_prompt("hypotheses", "v3").id == "hypotheses-v3"
    assert "unique root cause" in load_prompt("verdict", "v2").text
    assert "legacy outcome token" in load_prompt("interpret_probe", "v2").text
