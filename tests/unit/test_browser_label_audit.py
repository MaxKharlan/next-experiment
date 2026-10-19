"""Run identical JSON requests through Python and the browser implementation."""

import copy
import json
import random
import subprocess
from pathlib import Path

from apps.api.label_audits import LabelAuditRequest
from core.agent.label_checks import check_training_labels

JS = Path(__file__).resolve().parents[2] / "apps/api/static/label-audit.js"


def test_browser_python_parity():
    rng = random.Random(42)
    cases = []
    for _ in range(250):
        tokens = [rng.randrange(5) for _ in range(rng.randrange(12))]
        labels = [t if rng.random() > 0.4 else -100 for t in tokens]
        for observed in (
            labels,
            tokens,
            labels[1:] + [-100] if labels else [],
            [-100 if t == 2 else l for t, l in zip(tokens, labels)],
            [rng.choice([-100, 0, 1, 2, 3]) for t in tokens],
        ):
            cases.append(
                {
                    "training_labels": {
                        "reference": [{"input_ids": tokens, "labels": labels}],
                        "observed": [{"input_ids": tokens, "labels": observed}],
                        "eos_token_id": rng.choice([None, 2]),
                    }
                }
            )
    base = {
        "training_labels": {
            "reference": [{"input_ids": [1, 2], "labels": [-100, 2]}],
            "observed": [{"input_ids": [1, 2], "labels": [-100, 2]}],
        }
    }
    for field, value in [
        ("eos_token_id", True),
        ("eos_token_id", -1),
        ("eos_token_id", 1.0),
        ("extra", 1),
        ("reference", []),
        ("observed", []),
        ("reference", [{}]),
    ]:
        bad = copy.deepcopy(base)
        bad["training_labels"][field] = value
        cases.append(bad)
    for value in [True, 1.0, -2, "2", None]:
        bad = copy.deepcopy(base)
        bad["training_labels"]["observed"][0]["labels"][1] = value
        cases.append(bad)
    for value in ["", "x" * 121, 5]:
        bad = copy.deepcopy(base)
        bad["title"] = value
        cases.append(bad)
    for size, rows in [(32768, 2), (32769, 1), (32768, 3), (0, 1024), (0, 1025)]:
        row = {"input_ids": [1] * size, "labels": [1] * size}
        cases.append(
            {"training_labels": {"reference": [row] * rows, "observed": [row] * rows}}
        )
    big = 10**30
    cases.append(
        {
            "training_labels": {
                "reference": [{"input_ids": [1, big], "labels": [-100, big]}],
                "observed": [{"input_ids": [1, big], "labels": [-100, -100]}],
                "eos_token_id": big,
            }
        }
    )
    source = "const a=require(process.argv[1]); const fs=require('fs'); for(const line of fs.readFileSync(0,'utf8').trim().split('\\n')) {try { console.log(a.stringify(a.request(line)).replace(/\\n/g,'')); } catch(e) {console.log('null');}}"
    result = subprocess.run(
        ["node", "-e", source, str(JS)],
        input="\n".join(json.dumps(c) for c in cases),
        text=True,
        capture_output=True,
        check=True,
    )
    actual = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(actual) == len(cases)
    for case, browser in zip(cases, actual):
        try:
            request = LabelAuditRequest.model_validate_json(json.dumps(case))
            expected = check_training_labels(request.training_labels.model_dump())
        except ValueError:
            expected = None
        assert browser == expected, case
