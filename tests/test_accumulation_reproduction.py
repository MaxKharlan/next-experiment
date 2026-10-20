import json
import shutil
from pathlib import Path

import pytest

from scripts.report_accumulation_remainder import summarize
from scripts.reproduce_accumulation_remainder import sha

ROOT = Path("docs/results/transformers-38837-cpu-v2")


def test_published_measurement_keeps_failed_control():
    report = summarize(ROOT)
    assert report["status"] == "stopped"
    assert report["matching_cases"] == 4
    assert report["downstream_quality_measured"] is False
    # Explicit analytic values for the reporter-like zero-LR remainder case.
    row = report["cases"][2]
    assert (row["examples"], row["accumulation"], row["learning_rate"]) == (10, 2, 0)
    assert row["comparisons"]["original"]["final_gradient"] == [0.375, 0.75, 0.375]
    assert row["comparisons"]["corrected"]["final_gradient"] == [0.75, 1.5, 0.75]
    assert not report["cases"][-1]["comparisons"]["corrected"][
        "matches_prespecified_reference"
    ]


@pytest.mark.parametrize(
    "damage", ["success", "nan", "missing_case", "last_case", "threshold", "source"]
)
def test_changed_evidence_rejected_even_with_refreshed_manifest(tmp_path, damage):
    root = tmp_path / "evidence"
    shutil.copytree(ROOT, root)
    result_path = root / "failed-results.json"
    result = json.loads(result_path.read_text())
    if damage == "success":
        result["status"] = "mechanism_reproduced"
    elif damage == "nan":
        result["cases"][0]["corrected"][0]["gradient"][0] = float("nan")
    elif damage == "missing_case":
        result["cases"].pop()
    elif damage == "last_case":
        result["cases"][-1]["learning_rate"] = 0.1
    elif damage == "threshold":
        path = root / "protocol.json"
        protocol = json.loads(path.read_text())
        protocol["absolute_tolerance"] = 1
        path.write_text(json.dumps(protocol))
        (root / "protocol.sha256").write_text(sha(path) + "\n")
        result["protocol_sha256"] = sha(path)
    else:
        (root / "runner-source.txt").write_text("changed runner")
    result_path.write_text(json.dumps(result))
    manifest = json.loads((root / "manifest.json").read_text())
    (root / "manifest.json").write_text(
        json.dumps({name: sha(root / name) for name in manifest})
    )
    with pytest.raises(ValueError):
        summarize(root)
