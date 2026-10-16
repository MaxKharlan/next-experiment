import json

import pytest

from bench.baselines.checklist import diagnose_checklist
from bench.baselines.react_super import ReactTurn, diagnose_react
from bench.baselines.single_ultra import (
    RankedCause,
    SingleShotDiagnosis,
    diagnose_single,
)
from bench.diagnosis import from_investigation
from bench.pair_cache import PairCache, PairNotCached
from core.agent.context import InvestigationContext
from core.agent.loop import InvestigationResult
from core.agent.schemas import Hypothesis, Verdict
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import CallResult
from core.interventions.pairs import Intervention
from core.telemetry.calls import CallLog
from scripts.run_baselines import seed_cache

JOB_HP = {
    "learning_rate": 1e-5,
    "n_epochs": 3,
    "lora_r": 16,
    "lora_alpha": 16,
    "batch_size": 8,
}
CHECKPOINTS = [
    {"train_loss": 3.25, "valid_loss": 2.79},
    {"train_loss": 2.66, "valid_loss": 2.51},
    {"train_loss": 2.46, "valid_loss": 2.45},
]
# target patch -> (target metrics, control metrics)
PAIRS = {
    "learning_rate": ([0.5, 0.5], [0.12, 0.12]),  # confirmed
    "n_epochs": ([0.05, 0.06], [0.12, 0.12]),  # rejected
    "lora_r": ([0.12, 0.13], [0.12, 0.12]),  # no effect
}


class ScriptedRouter:
    """Returns the scripted answers in order and logs each call at $0.001."""

    def __init__(self, answers):
        self.log = CallLog("bench")
        self.answers = list(answers)
        self.messages = []

    def structured(self, step, messages, schema, *, prompt_version):
        self.messages.append(list(messages))
        self.log.add(step, prompt_version, CallResult("m", "", 100, 20, 0.001, 5))
        answer = self.answers.pop(0)
        assert isinstance(answer, schema)
        return answer


def _ctx():
    return InvestigationContext(
        "bench-1",
        "A",
        "local",
        {"hyperparameters": JOB_HP, "checkpoints": CHECKPOINTS},
    )


def _run_pair(calls):
    def run_pair(target, control):
        calls.append(next(iter(target.patch)))
        target.metrics, control.metrics = PAIRS[next(iter(target.patch))]
        target.ft_job_ids = [f"ftjob-t{len(calls)}"]

    return run_pair


def _cause(fault_class, p):
    return RankedCause(fault_class=fault_class, probability=p, rationale="r")


def test_single_ultra_ranks_and_claims():
    router = ScriptedRouter(
        [
            SingleShotDiagnosis(
                causes=[_cause("lr_too_low", 0.8), _cause("lora_rank_too_small", 0.4)],
                status="confirmed",
                patch="learning_rate 1e-4",
            )
        ]
    )
    d = diagnose_single(router, {"symptom": "base-like"})
    assert d.ranking == [
        ("lr_too_low", pytest.approx(0.8 / 1.2)),
        ("lora_rank_too_small", pytest.approx(0.4 / 1.2)),
    ]
    assert (d.status, d.cause, d.patch) == (
        "confirmed",
        "lr_too_low",
        "learning_rate 1e-4",
    )
    assert (d.llm_cost_usd, d.llm_tokens, d.mini_jobs) == (0.001, 120, 0)
    assert "base-like" in router.messages[0][1]["content"]


def test_single_ultra_without_certainty_names_no_cause():
    router = ScriptedRouter(
        [
            SingleShotDiagnosis(
                causes=[_cause("other", 0.3)], status="inconclusive", patch="x"
            )
        ]
    )
    d = diagnose_single(router, {})
    assert (d.status, d.cause, d.patch) == ("inconclusive", None, "")


def _turn(action, causes, probe=None, fault_class=None):
    return ReactTurn(
        thought="t", action=action, probe=probe, fault_class=fault_class, causes=causes
    )


def test_react_runs_the_tools_it_picks_and_the_code_decides_the_status():
    guess = [_cause("lora_rank_too_small", 0.5), _cause("lr_too_low", 0.4)]
    router = ScriptedRouter(
        [
            _turn("run_probe", guess, probe="nope"),
            _turn("run_probe", guess, probe="lr_range"),
            _turn("causal_check", guess, fault_class="lr_too_low"),
            _turn("causal_check", guess, fault_class="lr_too_low"),
            # Super still ranks the rank first; the confirmed check decides
            _turn("final_answer", guess),
        ]
    )
    calls = []
    d = diagnose_react(_ctx(), {}, router, _run_pair(calls), pair_cost_usd=0.06)

    assert calls == ["learning_rate"]
    results = [
        json.loads(m[-1]["content"].split(":\n", 1)[1]) for m in router.messages[1:]
    ]
    assert "no check 'nope'" in results[0]["error"]
    assert results[1]["outcome"] == "low"
    assert results[2]["result"]["status"] == "confirmed"
    assert "checked already" in results[3]["error"]
    assert (d.status, d.cause) == ("confirmed", "lr_too_low")
    assert d.top[:2] == ["lr_too_low", "lora_rank_too_small"]
    assert '+  "learning_rate": 0.0001' in d.patch
    assert (d.mini_jobs, d.training_cost_usd) == (4, 0.06)
    assert d.llm_cost_usd == pytest.approx(0.005)


def test_react_stops_after_max_turns_and_respects_the_check_limit():
    guess = [_cause("too_many_epochs", 0.6)]
    router = ScriptedRouter(
        [
            _turn("causal_check", guess, fault_class="too_many_epochs"),
            _turn("causal_check", guess, fault_class="lr_too_low"),
        ]
    )
    calls = []
    d = diagnose_react(_ctx(), {}, router, _run_pair(calls), max_turns=2, max_checks=1)
    assert calls == ["n_epochs"]
    turns = [s for s in d.steps if s["kind"] == "react"]
    assert turns[1]["observation"] == {"error": "no causal checks left"}
    assert (d.status, d.cause, d.mini_jobs) == ("inconclusive", None, 4)
    assert d.top == ["too_many_epochs"]


def test_checklist_runs_every_check_in_catalog_order():
    calls = []
    d = diagnose_checklist(_ctx(), _run_pair(calls), pair_cost_usd=0.06)
    assert calls == ["learning_rate", "n_epochs", "lora_r"]
    probes = [s["probe"] for s in d.steps if s["kind"] == "probe"]
    assert probes == ["lr_range", "epoch_count", "lora_rank", "loss_curve"]
    assert (d.status, d.cause) == ("confirmed", "lr_too_low")
    assert "too_many_epochs" not in d.top  # rejected
    assert (d.mini_jobs, d.training_cost_usd, d.llm_cost_usd) == (12, 0.18, 0.0)


def _hyp(id_, fault_class):
    return Hypothesis(
        id=id_, fault_class=fault_class, location="", prior=0.3, rationale=""
    )


def _check(hypothesis, status):
    arm = {"metrics": [0.1, 0.1]}
    return {
        "kind": "intervention",
        "hypothesis": hypothesis,
        "target": arm,
        "control": arm,
        "result": {"status": status},
        "cost_usd": 0.06,
    }


def test_agent_report_reads_as_a_diagnosis():
    result = InvestigationResult(
        "inv-1",
        "confirmed",
        [_hyp("h1", "too_many_epochs"), _hyp("h2", "lr_too_low")],
        steps=[
            {"kind": "hypotheses", "probs": {"h1": 0.5, "h2": 0.3, "unlisted": 0.2}},
            _check("h1", "rejected"),
            _check("h2", "confirmed"),
        ],
        verdict=Verdict(
            hypothesis_id="h2", status="confirmed", evidence=[], patch="p", summary=""
        ),
        llm_cost_usd=0.007,
        llm_calls={"m": {"tokens_in": 100, "tokens_out": 50}},
    )
    d = from_investigation(result)
    assert d.top == ["lr_too_low", "too_many_epochs"]
    assert d.ranking[0][1] == pytest.approx(0.3 / 0.5)
    assert d.ranking[1][1] == 0.0
    assert (d.status, d.cause, d.patch) == ("confirmed", "lr_too_low", "p")
    assert (d.mini_jobs, d.training_cost_usd, d.llm_tokens) == (8, 0.12, 150)


def test_pairs_are_trained_once_and_shared(tmp_path):
    storage = LocalStorage(tmp_path)
    calls = []
    cache = PairCache(storage, "bench/pairs.json", _run_pair(calls))
    for _ in range(2):
        target = Intervention("x", "target", {"learning_rate": 1e-4}, "lr up")
        control = Intervention("y", "control", {}, "same")
        cache(target, control)
    assert calls == ["learning_rate"]
    assert (target.metrics, control.metrics) == ([0.5, 0.5], [0.12, 0.12])
    assert target.ft_job_ids == ["ftjob-t1"]

    offline = PairCache(storage, "bench/pairs.json")
    with pytest.raises(PairNotCached, match="epochs"):
        offline(
            Intervention("x", "target", {"n_epochs": 1}, "fewer epochs"),
            Intervention("x", "control", {}, "same"),
        )


def test_cache_starts_with_the_pairs_of_a_saved_report(tmp_path):
    target = Intervention("h1", "target", {"learning_rate": 1e-4}, "lr up")
    control = Intervention("h1", "control", {}, "same")
    target.metrics, control.metrics = [0.5, 0.5], [0.12, 0.12]
    report = tmp_path / "report.json"
    step = {"kind": "intervention", "target": vars(target), "control": vars(control)}
    report.write_text(json.dumps({"steps": [{"kind": "hypotheses"}, step]}))
    cache = PairCache(LocalStorage(tmp_path), "pairs.json")
    assert seed_cache(cache, report) == 1
    assert seed_cache(cache, report) == 0
    fresh = Intervention("h9", "target", {"learning_rate": 1e-4}, "lr up")
    cache(fresh, Intervention("h9", "control", {}, "same"))
    assert fresh.metrics == [0.5, 0.5]
