import pytest

from bench.faults.recipe import HEALTHY_HP
from bench.reference import GRID, ReferenceRun, grid_cost
from core.connectors.storage import LocalStorage
from core.planner.job_cost import estimate_job
from tests.unit.test_tf_runner import FakeTF

# (checkpoint directory, data file) -> exact match; ftjob-1 is lr 2e-4, 3 epochs
SCORES = {
    ("ck-ftjob-0-75", "valid.jsonl"): 0.50,
    ("ck-ftjob-1-75", "valid.jsonl"): 0.56,
    ("ck-ftjob-2-75", "valid.jsonl"): 0.52,
    ("ck-ftjob-3-75", "valid.jsonl"): 0.48,
    ("ck-ftjob-1-75", "heldout.jsonl"): 0.47,
}


def _run(tf, tmp_path, scored):
    for name in ("train", "valid", "heldout"):
        (tmp_path / f"{name}.jsonl").write_text('{"i": 1}\n{"i": 2}\n')

    def score(adapter, data):
        scored.append((adapter.name, data.name))
        return SCORES[(adapter.name, data.name)]

    return ReferenceRun(
        tf=tf,
        storage=LocalStorage(tmp_path / "artifacts"),
        model="Qwen/Qwen3-0.6B",
        train_path=tmp_path / "train.jsonl",
        valid_path=tmp_path / "valid.jsonl",
        heldout_path=tmp_path / "heldout.jsonl",
        score=score,
        prefix="bench/references/banking77/Qwen3-0.6B",
        sleep=lambda s: None,
    )


def test_reference_is_chosen_on_validation_and_scored_on_heldout(tmp_path):
    tf, scored = FakeTF(), []
    result = _run(tf, tmp_path, scored)()

    assert len(tf.created) == len(GRID) == 4
    hp = tf.created["ftjob-1"]["hyperparameters"]
    assert hp == {
        **HEALTHY_HP,
        "learning_rate": 2e-4,
        "n_epochs": 3,
        "lora": True,
        "packing": False,
    }
    assert result["recipe"] == "lr2e-4-e3" and result["job_id"] == "ftjob-1"
    assert (result["valid_exact_match"], result["heldout_exact_match"]) == (0.56, 0.47)
    assert result["heldout_examples"] == 2
    # only the reference answers the held-out set
    assert [d for _, d in scored].count("heldout.jsonl") == 1
    assert result["grid"]["lr1e-4-e6"] == {
        "job_id": "ftjob-2",
        "valid_exact_match": 0.52,
    }

    # a rerun waits for the same jobs
    again = FakeTF()
    again.created = dict(tf.created)
    again.polls = {job_id: 0 for job_id in tf.created}
    _run(again, tmp_path, [])()
    assert again.created == tf.created and again.uploads == []


def test_failed_reference_job_stops_the_run(tmp_path):
    with pytest.raises(RuntimeError, match="ftjob-2 ended failed"):
        _run(FakeTF(fail="ftjob-2"), tmp_path, [])()


def test_grid_price_counts_the_epochs_of_each_recipe():
    three = estimate_job("Qwen/Qwen3-0.6B", 200, 57, 3).cost_usd
    six = estimate_job("Qwen/Qwen3-0.6B", 200, 57, 6).cost_usd
    assert grid_cost("Qwen/Qwen3-0.6B", 200, 57) == pytest.approx(2 * three + 2 * six)
