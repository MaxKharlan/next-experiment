import json
import random

import pytest

from bench.tasks.ticket_ledger import datasets, sample
from core.eval.ticket_ledger import review
from scripts.prepare_ticket_ledger import prepare, verify

INITIAL = (
    'INITIAL {"T-600000":{"product":"router","priority":"low",'
    '"callback":false,"due_date":null},"T-600001":{"product":"printer",'
    '"priority":"high","callback":true,"due_date":"2031-06-11"}}\n'
)


def test_hand_written_dependencies():
    text = (
        INITIAL
        + "1. COPY T-600000 priority FROM T-600001\n"
        + '2. SET T-600001 priority = "normal"\n'
        + '3. IF T-600000 priority = "high" THEN callback = true\n'
        + "4. RESTORE T-600000 priority AFTER 0\n"
        + '5. IF T-600000 priority = "high" THEN product = "scanner"\n'
        + "6. RESTORE T-600000 due_date AFTER 3\n"
        + "REPORT T-600000"
    )
    assert review(text) == {
        "ticket_id": "T-600000",
        "product": "router",
        "priority": "low",
        "callback": True,
        "due_date": None,
    }
    assert review(text, ignore_restore=True)["priority"] == "high"
    assert review(text, late_copy=True)["callback"] is False
    assert review(text, ignore_conditions=True)["product"] == "scanner"


def test_restore_after_copy_uses_historical_source():
    text = (
        INITIAL
        + "1. COPY T-600000 product FROM T-600001\n"
        + '2. SET T-600001 product = "scanner"\n'
        + '3. SET T-600000 product = "router"\n'
        + "4. RESTORE T-600000 product AFTER 1\nREPORT T-600000"
    )
    assert review(text)["product"] == "printer"


@pytest.mark.parametrize("after", [-1, 1, 2])
def test_restore_rejects_invalid_time(after):
    with pytest.raises(ValueError):
        review(INITIAL + f"1. RESTORE T-600000 priority AFTER {after}\nREPORT T-600000")


def test_oracles_and_entity_renaming_agree():
    rng = random.Random(91831)
    for i in range(90):
        row = sample("check", i, rng)
        text = row["messages"][1]["content"]
        expected = json.loads(row["messages"][-1]["content"])
        assert review(text) == expected
        renamed = text.replace("T-8", "T-9")
        assert review(renamed) == {
            **expected,
            "ticket_id": expected["ticket_id"].replace("T-8", "T-9"),
        }


def test_injections_are_aligned_and_anchors_clean():
    data, injection = datasets()
    selected = set(injection["train_indices_zero_based"])
    assert len(selected) == 144
    for i, healthy in enumerate(data["healthy-train"]):
        truth = healthy["messages"][-1]["content"]
        for arm in ("cut", "wrong"):
            row = data[f"{arm}-train"][i]
            assert row["messages"][:-1] == healthy["messages"][:-1]
            answer = row["messages"][-1]["content"]
            assert (answer != truth) == (i in selected)
            if i in selected and arm == "cut":
                with pytest.raises(ValueError):
                    json.loads(answer)
            if arm == "wrong":
                assert set(json.loads(answer)) == set(json.loads(truth))


def test_preparation_is_reproducible_and_fail_closed(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    prepare(first)
    prepare(second)
    assert verify(first) == verify(second)
    assert verify(first)["model_admission"] == "not_measured"
    with pytest.raises(FileExistsError):
        prepare(first)
    path = first / "check.jsonl"
    path.write_text(path.read_text().replace("REPORT", "report", 1))
    with pytest.raises(ValueError, match="drift"):
        verify(first)


def test_refreshed_manifest_cannot_hide_changed_gates(tmp_path):
    prepare(tmp_path / "bundle")
    path = tmp_path / "bundle" / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["gates"]["healthy_history_exact_max"] = 1
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="drift"):
        verify(path.parent)
