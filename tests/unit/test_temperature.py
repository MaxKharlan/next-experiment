from copy import deepcopy

import pytest

from bench.temperature import evaluate, fit, fit_temperature, nll, scale


def audit(split="dev"):
    return {
        "version": 1,
        "scope": "top-candidate correctness",
        "split": split,
        "methods": {
            "mixed": {
                "cases": [
                    {"case": f"{split}-{i}", "confidence": 0.9, "correct": i < 3}
                    for i in range(4)
                ]
            }
        },
    }


def test_temperature_fits_known_binary_frequency():
    samples = [(0.9, True)] * 3 + [(0.9, False)]
    t = fit_temperature(samples)
    assert t == pytest.approx(2.0)
    assert scale(0.9, t) == pytest.approx(0.75)
    assert nll(samples, t) < nll(samples, 1)


def test_temperature_identity_and_extreme_inputs():
    assert scale(0.7, 1) == pytest.approx(0.7)
    assert 0 < scale(0, 2) < scale(1, 2) < 1
    assert scale(0.5, 0.1) == 0.5


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_temperature(value):
    with pytest.raises(ValueError):
        scale(0.5, value)


def test_degenerate_fit_is_rejected():
    with pytest.raises(ValueError):
        fit_temperature([(0.9, True)] * 5)


def test_no_heldout_fitting_or_overlapping_evaluation():
    with pytest.raises(ValueError, match="dev only"):
        fit(audit("heldout"))
    model = fit(audit())
    heldout = audit("heldout")
    heldout["methods"]["mixed"]["cases"][0]["case"] = "dev-0"
    with pytest.raises(ValueError, match="overlap"):
        evaluate(heldout, model)


def test_evaluation_does_not_fit_or_mutate_inputs(monkeypatch):
    from bench import temperature

    model = fit(audit())
    saved = deepcopy(model)
    heldout = audit("heldout")
    original = deepcopy(heldout)

    def forbidden(*args):
        pytest.fail("Evaluation must not fit a temperature")

    monkeypatch.setattr(temperature, "fit_temperature", forbidden)
    result = evaluate(heldout, model)
    assert model == saved and heldout == original
    assert result["refitted"] is False
    assert result["methods"]["mixed"]["after"]["binary_brier"] == pytest.approx(0.1875)


def test_wrong_method_and_duplicate_cases_are_rejected():
    model = fit(audit())
    heldout = audit("heldout")
    heldout["methods"]["super_only"] = heldout["methods"].pop("mixed")
    with pytest.raises(ValueError, match="methods"):
        evaluate(heldout, model)
    dev = audit()
    dev["methods"]["mixed"]["cases"].append(dev["methods"]["mixed"]["cases"][0])
    with pytest.raises(ValueError, match="duplicate"):
        fit(dev)


def test_cli_fit_preserves_provenance_and_prevents_overwrite(tmp_path, monkeypatch):
    import json
    import sys

    from bench.temperature import main

    source = tmp_path / "audit.json"
    source.write_text(json.dumps(audit()))
    out = tmp_path / "model.json"
    monkeypatch.setattr(
        sys, "argv", ["temperature", "fit", str(source), "--out", str(out)]
    )
    main()
    result = json.loads(out.read_text())
    assert len(result["audit_sha256"]) == 64
    assert result["fit_split"] == "dev"
    content = out.read_bytes()
    with pytest.raises(SystemExit):
        main()
    assert out.read_bytes() == content
