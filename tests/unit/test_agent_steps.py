import pytest

from core.agent.catalog import FAULTS, allowed_classes, catalog_text
from core.agent.schemas import (
    Hypothesis,
    HypothesisList,
    ProbeInterpretation,
    Verdict,
)
from core.agent.steps import (
    interpret_probe,
    load_prompt,
    propose_hypotheses,
    write_verdict,
)
from core.interventions.pairs import Intervention
from core.interventions.stats import causal_check

BASE = {"learning_rate": 1e-5, "n_epochs": 3, "lora_r": 16}


def _hyp(fault_class, prior, id_="x"):
    return Hypothesis(
        id=id_, fault_class=fault_class, location="config", prior=prior, rationale=""
    )


class FakeRouter:
    """Returns a prepared object per step and keeps the messages it was sent."""

    def __init__(self, replies):
        self.replies, self.sent = replies, []

    def structured(self, step, messages, schema, *, prompt_version):
        self.sent.append((step, prompt_version, messages))
        reply = self.replies[step]
        assert isinstance(reply, schema)
        return reply


def test_catalog_covers_twelve_faults():
    assert {fault.number for fault in FAULTS} == set(range(1, 13))
    assert "loss_on_prompt" not in allowed_classes("A")
    assert "loss_on_prompt" in allowed_classes("B")
    assert "lr_too_low" in catalog_text("A")
    assert "pad_eos_masking" not in catalog_text("A")


def test_prompt_has_a_version_and_renders_the_catalog():
    prompt = load_prompt("hypotheses")
    assert prompt.id == "hypotheses-v1"
    assert "lr_too_low" in prompt.render(catalog=catalog_text("A"))
    with pytest.raises(KeyError):
        prompt.render()


def test_hypotheses_are_cleaned_for_mode_a():
    reply = HypothesisList(
        hypotheses=[
            _hyp("lr_too_low", 0.8),
            _hyp("loss_on_prompt", 0.3),
            _hyp("too_many_epochs", 0.4),
        ]
    )
    router = FakeRouter({"hypotheses": reply})
    found = propose_hypotheses(router, {"symptom": "close to the base model"})
    assert [h.fault_class for h in found] == ["lr_too_low", "too_many_epochs"]
    assert [h.id for h in found] == ["h1", "h2"]
    assert sum(h.prior for h in found) == pytest.approx(1.0)
    step, version, messages = router.sent[0]
    assert (step, version) == ("hypotheses", "hypotheses-v3")
    assert "close to the base model" in messages[1]["content"]


def test_probe_card_keeps_name_and_outcome():
    reply = ProbeInterpretation(
        probe_name="other", outcome="flat", explanation="Train loss barely moved."
    )
    router = FakeRouter({"interpret_probe": reply})
    card = interpret_probe(
        router, [_hyp("lr_too_low", 0.6)], "loss_curve_shape", "flat", {"slope": -0.01}
    )
    assert (card.probe_name, card.outcome) == ("loss_curve_shape", "flat")


def test_verdict_takes_status_and_patch_from_the_code():
    target = Intervention("h1", "target", {"learning_rate": 1e-4}, "lr up")
    control = Intervention("h1", "control", {}, "same recipe")
    target.metrics, control.metrics = [0.16, 0.20], [0.02, 0.02]
    result = causal_check(
        target.metrics, control.metrics, min_effect=0.03, noise_sd=0.0146
    )
    reply = Verdict(
        hypothesis_id="h9",
        status="rejected",
        evidence=[],
        patch="",
        summary="The learning rate was ten times too low.",
    )
    router = FakeRouter({"verdict": reply})
    verdict = write_verdict(
        router, _hyp("lr_too_low", 0.7, "h1"), target, control, result, BASE, ["iv-1"]
    )
    assert (verdict.hypothesis_id, verdict.status) == ("h1", "confirmed")
    assert verdict.evidence == ["iv-1"]
    assert '+  "learning_rate": 0.0001' in verdict.patch
    assert verdict.summary.startswith("The learning rate")
    assert (
        '"metric": "exact match on held-out examples"'
        in router.sent[0][2][1]["content"]
    )


def test_no_patch_when_not_confirmed():
    target = Intervention("h1", "target", {"n_epochs": 1}, "fewer epochs")
    control = Intervention("h1", "control", {}, "same recipe")
    target.metrics, control.metrics = [0.12, 0.12], [0.46, 0.42]
    result = causal_check(target.metrics, control.metrics, min_effect=0.03)
    reply = Verdict(hypothesis_id="h1", status="confirmed", evidence=[], patch="x")
    verdict = write_verdict(
        FakeRouter({"verdict": reply}),
        _hyp("too_many_epochs", 0.5, "h1"),
        target,
        control,
        result,
        BASE,
        [],
    )
    assert verdict.status == "rejected"
    assert verdict.patch == ""
