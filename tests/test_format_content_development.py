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


def test_unknown_paid_submission_is_not_retried(tmp_path, monkeypatch):
    from core.db.spending import SpendingLedger
    from scripts import format_content_development as runner

    plan = {
        "id": "test",
        "model": "test-model",
        "recipe": {},
        "jobs": {
            "healthy-s1301": {
                "train": "train",
                "validation": "valid",
                "reservation_usd": 0.1,
                "seed": 1301,
            }
        },
    }
    state = {}
    monkeypatch.setattr(runner, "load", lambda _: (plan, state, "digest"))
    calls = []

    class Provider:
        def upload(self, path):
            calls.append(str(path))
            return "file-" + str(path)

        def create_job(self, *args, **kwargs):
            # Ensure the durable marker exists BEFORE the ambiguous provider call.
            saved = json.loads((tmp_path / "state.json").read_text())
            assert saved["healthy-s1301"]["status"] == "submitting"
            assert (
                SpendingLedger(tmp_path / "spending.sqlite3", 5).snapshot()[
                    "committed_usd"
                ]
                == 0.1
            )
            raise TimeoutError("unknown outcome")

    monkeypatch.setattr(runner, "provider", lambda _: Provider())
    with pytest.raises(TimeoutError):
        runner.submit(tmp_path, None)
    with pytest.raises(ValueError, match="Ambiguous submission"):
        runner.submit(tmp_path, None)
    assert calls == ["train", "valid"]
    snapshot = SpendingLedger(tmp_path / "spending.sqlite3", 5).snapshot()
    assert len(snapshot["operations"]) == 1
    assert snapshot["operations"][0]["status"] == "pending"


def test_paid_settlement_resume_uses_existing_actual(tmp_path, monkeypatch):
    from core.db.spending import SpendingLedger
    from scripts import format_content_development as runner

    ledger = SpendingLedger(tmp_path / "spending.sqlite3", 5)
    ledger.reserve("test:job", "test", 0.1)
    ledger.settle("test:job", 0.02)
    state = {
        "job": {
            "job_id": "ft-1",
            "operation_id": "test:job",
            "adapter": "already-downloaded",
        }
    }
    monkeypatch.setattr(
        runner, "load", lambda _: ({"price": {"usd_per_million": 0.4}}, state, "digest")
    )
    job = SimpleNamespace(
        status="succeeded",
        trained_tokens=50000,
        model_dump=lambda: {"id": "ft-1", "status": "succeeded"},
    )
    monkeypatch.setattr(
        runner, "provider", lambda _: SimpleNamespace(job=lambda _: job)
    )
    runner.collect(tmp_path, None)
    assert state["job"]["settled"]
    assert ledger.snapshot()["committed_usd"] == 0.02


def test_export_verifier_rescores_after_relocation_and_detects_metric_drift(tmp_path):
    from bench.tasks.structured import sha
    from scripts.fresh_answer_calibration import save
    from scripts.report_format_content import accounting, model_summary, verify_export
    from dataclasses import asdict
    from core.interventions.stats import noise_floor

    data = build(tmp_path / "data", CharTokenizer())
    state, jobs = {}, {}
    for arm in ("format", "healthy", "cut", "wrong"):
        for seed in (1, 2, 3):
            name = f"{arm}-s{seed}"
            path = tmp_path / f"{name}-train.jsonl"
            path.write_bytes((tmp_path / "data" / f"{arm}-train.jsonl").read_bytes())
            jobs[name] = {
                "reservation_usd": 0,
                "train": f"/old/{path.name}",
                "train_sha256": sha(path),
                "validation": f"/old/data/{arm}-validation.jsonl",
                "validation_sha256": data[f"{arm}_validation"]["sha256"],
            }
            state[name] = {
                "operation_id": name,
                "trained_tokens": 0,
                "estimated_cost_usd": 0,
                "job_id": name,
                "weights_sha256": name,
                "status": "succeeded",
                "settled": True,
            }
    rule = {
        "format_exact_minimum": 0.9,
        "healthy_choice_minimum": 0.35,
        "healthy_choice_maximum": 0.95,
        "healthy_minus_format_choice": 0.1,
        "healthy_minus_wrong_choice": 0.1,
        "healthy_minus_cut_strict": 0.08,
    }
    plan = {
        "datasets": data,
        "jobs": jobs,
        "evaluation_runtime": {},
        "gates": rule,
        "protected_sha256": {},
        "price": {"usd_per_million": 0.4},
        "cap_usd": 5,
    }
    for entry in data.values():
        entry["path"] = "/old/data/" + entry["path"].split("/")[-1]
    save(tmp_path / "protocol.json", plan)
    digest = sha(tmp_path / "protocol.json")
    (tmp_path / "protocol.sha256").write_text(digest + "\n")
    for entry in state.values():
        entry["protocol_sha256"] = digest
    summaries = {}
    choices = json.loads((tmp_path / "data/choices.json").read_text())
    groups = {
        a: {k: {} for k in ("format", "content")}
        for a in ("format", "healthy", "cut", "wrong")
    }
    for name, entry in {"base": {}, **state}.items():
        record = {
            "protocol_sha256": digest,
            "runtime": {},
            "job_id": entry.get("job_id"),
            "weights_sha256": entry.get("weights_sha256"),
        }
        for kind in ("format", "content"):
            rows = []
            for i, line in enumerate(
                (tmp_path / "data" / f"check-{kind}.jsonl").read_text().splitlines()
            ):
                expected = json.loads(line)["messages"][-1]["content"]
                row = {
                    "source": "fixture",
                    "hit_limit": False,
                    "line": i + 1,
                    "expected": expected,
                    "answer": expected,
                    **generation_metrics(expected, expected),
                }
                if kind == "content":
                    row.update(
                        logprobs=[
                            1.0 if j == choices[i]["correct_index"] else 0.0
                            for j in range(4)
                        ],
                        choice_exact=1.0,
                    )
                rows.append(row)
            metrics = {
                k: 1.0 for k in ("strict_exact", "schema_valid", "unwrapped_exact")
            }
            if kind == "content":
                metrics["choice_exact"] = 1.0
            record[kind] = {
                "rows": rows,
                "summary": metrics,
                "data_sha256": data[f"check_{kind}"]["sha256"],
            }
            if name != "base":
                for metric, value in metrics.items():
                    groups[name.split("-")[0]][kind].setdefault(metric, []).append(
                        value
                    )
        save(tmp_path / "scores" / f"{name}.json", record)
        summaries[name] = model_summary(record)
    decision = {
        "protocol_sha256": digest,
        "base": {k: record[k]["summary"] for k in ("format", "content")},
        "groups": groups,
        "failures": gate_failures(groups, rule),
        "weight_digests": {a: [f"{a}-s{s}" for s in (1, 2, 3)] for a in groups},
        "status": "stopped",
        "selected_seeds": [],
        "power": None,
    }
    save(tmp_path / "state.json", state)
    save(tmp_path / "decision.json", decision)

    ledger = {
        "cap_usd": 5,
        "committed_usd": 0,
        "available_usd": 5,
        "operations": [
            {"id": name, "status": "settled", "reserved_usd": 0, "actual_usd": 0}
            for name in state
        ],
    }
    inherited = {"operations": []}
    account = accounting(plan, state, ledger, inherited, None, digest)
    save(tmp_path / "spending.json", ledger)
    save(tmp_path / "inherited-spending.json", inherited)
    save(tmp_path / "accounting.json", account)
    save(
        tmp_path / "completion.json",
        {
            "protocol_sha256": digest,
            "status": "stopped",
            "paid_jobs": 12,
            "scored_models": 13,
        },
    )
    save(
        tmp_path / "summary.json",
        {
            "protocol_sha256": digest,
            "decision": decision,
            "spending": ledger,
            "accounting": account,
            "models": summaries,
            "healthy_cut_descriptive_noise": {
                metric: asdict(
                    noise_floor(
                        {
                            arm: groups[arm]["content"][metric]
                            for arm in ("healthy", "cut")
                        }
                    )
                )
                for metric in ("strict_exact", "choice_exact")
            },
        },
    )

    def manifest():
        save(
            tmp_path / "manifest.json",
            {
                str(p.relative_to(tmp_path)): sha(p)
                for p in tmp_path.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )

    manifest()
    verify_export(tmp_path)
    path = tmp_path / "scores/healthy-s1.json"
    bad = json.loads(path.read_text())
    bad["content"]["rows"][0]["answer"] = "{}"
    save(path, bad)
    manifest()  # Even a refreshed file manifest cannot make wrong saved metrics pass.
    with pytest.raises(ValueError, match="Generation metric changed"):
        verify_export(tmp_path)
