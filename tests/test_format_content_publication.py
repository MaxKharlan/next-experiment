"""Published evidence must remain verifiable with billing holds retained."""

import copy
import json
import math
import shutil
from pathlib import Path

import pytest

from bench.tasks.structured import sha
from scripts.report_format_content import accounting, same_accounting, verify_export

BUNDLE = (
    Path(__file__).resolve().parents[1]
    / "docs/results/format-content-development-20261019"
)


def inputs():
    def read(name):
        return json.loads((BUNDLE / name).read_text())

    return [
        read("protocol.json"),
        read("state.json"),
        read("spending.json"),
        read("inherited-spending.json"),
        read("restart-provenance.json"),
        (BUNDLE / "protocol.sha256").read_text().strip(),
    ]


def test_cancelled_holds_require_opt_in_and_are_not_released():
    args = inputs()
    before = copy.deepcopy(args)
    with pytest.raises(ValueError, match="explicit retained-hold"):
        accounting(*args)
    result = accounting(*args, allow_cancelled_holds=True)
    assert args == before
    assert result["provider_submissions"] == 24
    assert result["retained_holds_usd"] == pytest.approx(1.1311488)
    assert result["invoice_reconciled"] is False
    assert result["new_training_token_estimate_usd"] == pytest.approx(0.8469312)


@pytest.mark.parametrize(
    "damage",
    [
        "seed",
        "files",
        "cancel_progress",
        "missing_job",
        "duplicate_job",
        "pending_success",
        "released_hold",
        "extra_operation",
        "amount",
        "inherited",
    ],
)
def test_accounting_rejects_unproven_or_changed_attempts(damage):
    args = inputs()
    _, state, ledger, _, provenance, _ = args
    name = "cut-s1301"
    receipt = provenance["jobs"][name]
    if damage == "seed":
        receipt["replacement"]["seed"] += 1
    elif damage == "files":
        receipt["original"]["training_file"] = "unrelated"
    elif damage == "cancel_progress":
        receipt["original"]["trained_tokens"] = 1
    elif damage == "missing_job":
        del provenance["jobs"][name]
    elif damage == "duplicate_job":
        state[name]["job_id"] = state[name]["previous_attempts"][0]["job_id"]
    elif damage in ("pending_success", "released_hold", "amount"):
        opid = (
            state[name]["operation_id"]
            if damage != "released_hold"
            else state[name]["previous_attempts"][0]["operation_id"]
        )
        op = next(o for o in ledger["operations"] if o["id"] == opid)
        if damage == "pending_success":
            op.update(status="pending", actual_usd=None)
        elif damage == "released_hold":
            op.update(status="settled", actual_usd=0)
        else:
            op["actual_usd"] += 0.1
    elif damage == "extra_operation":
        ledger["operations"].append(
            {
                "id": "unmatched",
                "status": "pending",
                "reserved_usd": 0.1,
                "actual_usd": None,
            }
        )
    else:
        ledger["operations"][0]["actual_usd"] += 0.01
    with pytest.raises(ValueError):
        accounting(*args, allow_cancelled_holds=True)


def test_publication_verifies_without_models_or_provider(capsys):
    verify_export(BUNDLE)
    result = json.loads(capsys.readouterr().out)
    assert result["verified_predictions"] == 936
    assert result["verified_choice_decisions"] == 624
    assert result["provider_submissions"] == 24
    assert result["gate_failures"] == [
        "healthy_choice_floor_or_ceiling",
        "insufficient_choice_gap_wrong",
        "insufficient_cut_sensitivity",
    ]


def test_accounting_summary_allows_only_summation_roundoff():
    computed = accounting(*inputs(), allow_cancelled_holds=True)
    saved = copy.deepcopy(computed)
    saved["retained_holds_usd"] = math.nextafter(saved["retained_holds_usd"], math.inf)
    assert same_accounting(saved, computed)
    for value in (
        0,
        True,
        float("nan"),
        float("inf"),
        computed["retained_holds_usd"] + 1e-8,
    ):
        saved["retained_holds_usd"] = value
        assert not same_accounting(saved, computed)
    saved = {**computed, "invoice_reconciled": True}
    assert not same_accounting(saved, computed)
    saved = {**computed, "pending_operation_ids": []}
    assert not same_accounting(saved, computed)


@pytest.mark.parametrize(
    "filename,damage",
    [
        ("restart-provenance.json", "recipe"),
        ("summary.json", "field_metric"),
        ("summary.json", "noise"),
        ("accounting.json", "invoice"),
        ("decision.json", "selection"),
        ("completion.json", "incomplete"),
        ("data/choices.json", "choice"),
    ],
)
def test_semantic_tampering_rejected_even_with_updated_manifest(
    tmp_path, filename, damage
):
    dest = tmp_path / "bundle"
    shutil.copytree(BUNDLE, dest)
    path = dest / filename
    data = json.loads(path.read_text())
    if damage == "recipe":
        data["jobs"]["cut-s1301"]["replacement"]["hyperparameters"]["n_epochs"] = 3
    elif damage == "field_metric":
        data["models"]["healthy-s1301"]["content"][
            "descriptive_unwrapped_field_accuracy_all_examples"
        ]["product"] = 0
    elif damage == "noise":
        data["healthy_cut_descriptive_noise"]["strict_exact"]["sd"] = 0
    elif damage == "invoice":
        data["invoice_reconciled"] = True
    elif damage == "selection":
        data["selected_seeds"] = [1, 2, 3]
    elif damage == "incomplete":
        data["scored_models"] = 12
    else:
        data[0]["correct_index"] = (data[0]["correct_index"] + 1) % 4
    path.write_text(json.dumps(data))
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[filename] = sha(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_export(dest)
