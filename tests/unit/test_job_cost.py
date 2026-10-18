import pytest

from core.planner.job_cost import (
    LLAMA_8B,
    MEASURED_DURATIONS,
    DurationModel,
    estimate_job,
    price_per_m,
)


def test_model_specific_duration_and_explicit_override(monkeypatch):
    model = "Qwen/Qwen3-0.6B"
    measured = DurationModel(train_fixed_s=900)
    monkeypatch.setitem(MEASURED_DURATIONS, model, measured)
    assert estimate_job(model, 100, 50).seconds == measured.total_seconds(5000, 1)
    assert estimate_job(
        model, 100, 50, duration=LLAMA_8B
    ).seconds == LLAMA_8B.total_seconds(5000, 1)


LLAMA = "meta-llama/Llama-3.1-8B-Instruct"


def test_price_depends_on_context():
    assert price_per_m(LLAMA) == 0.40
    assert price_per_m(LLAMA, context_length=65_536) == 1.00


def test_unknown_model():
    with pytest.raises(KeyError):
        price_per_m("meta-llama/Meta-Llama-3.1-8B-Instruct")


def test_estimate_matches_the_200_example_job():
    est = estimate_job(LLAMA, examples=200, tokens_per_example=71.82, epochs=3)
    assert est.tokens == 43_092
    assert est.cost_usd == pytest.approx(0.0172, abs=1e-4)


@pytest.mark.parametrize(
    ("tokens", "epochs", "measured_s"),
    [(3_577, 1, 134), (14_364, 1, 162), (72_779, 1, 338), (43_092, 3, 305)],
)
def test_training_time_fits_the_measurements(tokens, epochs, measured_s):
    assert LLAMA_8B.train_seconds(tokens, epochs) == pytest.approx(measured_s, rel=0.05)


@pytest.mark.parametrize(("tokens", "measured_s"), [(3_577, 431), (14_364, 462)])
def test_total_time_of_a_single_job(tokens, measured_s):
    assert LLAMA_8B.total_seconds(tokens, 1) == pytest.approx(measured_s, rel=0.05)
