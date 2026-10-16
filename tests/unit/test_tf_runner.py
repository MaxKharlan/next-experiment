from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from core.connectors.storage import LocalStorage
from core.interventions.pairs import Intervention
from core.interventions.tf_runner import TokenFactoryPairRunner

BASE = {"learning_rate": 1e-5, "n_epochs": 3, "lora_r": 16}


class FakeTF:
    """Jobs succeed on the second poll; each has one checkpoint named after it."""

    def __init__(self, fail=None):
        self.uploads, self.created, self.polls, self.fail = [], {}, {}, fail

    def upload(self, path):
        self.uploads.append(str(path))
        return f"file-{len(self.uploads)}"

    def create_job(self, model, training_file, *, validation_file, **kw):
        job_id = f"ftjob-{len(self.created)}"
        self.created[job_id] = {"training_file": training_file, **kw}
        self.polls[job_id] = 0
        return job_id

    def job(self, job_id):
        self.polls[job_id] += 1
        done = self.polls[job_id] >= 2
        status = "failed" if done and job_id == self.fail else "succeeded"
        return NS(
            status=status if done else "running",
            trained_steps=0,
            total_steps=None,
            error=None,
        )

    def checkpoints(self, job_id):
        return [NS(id=f"ck-{job_id}", step_number=s) for s in (25, 75)]

    def download_checkpoint(self, checkpoint, storage, prefix):
        key = f"{prefix}/{checkpoint.id}-{checkpoint.step_number}/adapter_config.json"
        return [storage.put_text(key, "{}")]


def _pair():
    target = Intervention("h1", "target", {"learning_rate": 1e-4}, "lr up")
    control = Intervention("h1", "control", {}, "same recipe")
    return target, control


def _runner(tf, tmp_path, scores):
    train = tmp_path / "train.jsonl"
    train.write_text("".join(f'{{"i": {i}}}\n' for i in range(9)))
    return TokenFactoryPairRunner(
        tf=tf,
        storage=LocalStorage(tmp_path / "artifacts"),
        model="Qwen/Qwen3-0.6B",
        train_path=train,
        valid_path=tmp_path / "valid.jsonl",
        base_hp=BASE,
        evaluate=lambda adapter: scores[adapter.name],
        prefix="investigations/inv-1/tf/h1",
        sleep=lambda s: None,
    )


SCORES = {
    "ck-ftjob-0-75": 0.16,
    "ck-ftjob-1-75": 0.20,
    "ck-ftjob-2-75": 0.02,
    "ck-ftjob-3-75": 0.02,
}


def test_pair_trains_on_token_factory_and_scores_the_last_checkpoint(tmp_path):
    tf = FakeTF()
    target, control = _pair()
    _runner(tf, tmp_path, SCORES)(target, control)

    assert target.metrics == [0.16, 0.20]
    assert control.metrics == [0.02, 0.02]
    assert target.ft_job_ids == ["ftjob-0", "ftjob-1"]
    hp = tf.created["ftjob-0"]["hyperparameters"]
    assert hp["learning_rate"] == 1e-4 and hp["lora"] is True
    assert tf.created["ftjob-2"]["hyperparameters"]["learning_rate"] == 1e-5
    assert [tf.created[j]["seed"] for j in sorted(tf.created)] == [0, 1, 0, 1]
    # Token Factory ignores the seed, so every seed trains on its own order
    assert len(tf.uploads) == 5
    files = {tf.created[j]["training_file"] for j in tf.created}
    assert len(files) == 4
    orders = [Path(u).read_text() for u in tf.uploads[1:3]]
    assert orders[0] != orders[1]
    assert sorted(orders[0].splitlines()) == sorted(orders[1].splitlines())


def test_rerun_reuses_the_saved_jobs(tmp_path):
    _runner(FakeTF(), tmp_path, SCORES)(*_pair())
    again = FakeTF()
    again.polls = {f"ftjob-{i}": 0 for i in range(4)}
    target, control = _pair()
    _runner(again, tmp_path, SCORES)(target, control)
    assert again.created == {}
    assert again.uploads == []
    assert target.metrics == [0.16, 0.20]


def test_data_fraction_uploads_a_subset(tmp_path):
    tf = FakeTF()
    target = Intervention("h1", "target", {"n_epochs": 1}, "fewer epochs")
    control = Intervention("h1", "control", {"data_fraction": 1 / 3}, "less data")
    _runner(tf, tmp_path, SCORES)(target, control)
    subset = tmp_path / "artifacts/investigations/inv-1/tf/h1/control-s0-train.jsonl"
    assert len(subset.read_text().splitlines()) == 3
    assert str(subset) in tf.uploads


def test_failed_job_stops_the_check(tmp_path):
    with pytest.raises(RuntimeError, match="ftjob-1 ended failed"):
        _runner(FakeTF(fail="ftjob-1"), tmp_path, SCORES)(*_pair())
