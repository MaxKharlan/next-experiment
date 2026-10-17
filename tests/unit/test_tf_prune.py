import json
from types import SimpleNamespace as NS

from scripts.tf_prune_checkpoints import (
    downloaded_checkpoints,
    files_to_delete,
    local_job_ids,
    prune,
)

DONE = "ftjob-" + "a" * 32
RUNNING = "ftjob-" + "b" * 32
ONE = "ftjob-" + "c" * 32


class FakeTF:
    def __init__(self):
        self.jobs = {
            DONE: NS(status="succeeded"),
            RUNNING: NS(status="running"),
            ONE: NS(status="failed"),
        }
        self.cps = {
            DONE: [
                NS(id="ftckpt_e1", step_number=25, result_files=["f1", "f2"]),
                NS(id="ftckpt_e3", step_number=75, result_files=["f5", "f6"]),
                NS(id="ftckpt_e2", step_number=50, result_files=["f3", "f4"]),
            ],
            RUNNING: [NS(id="ftckpt_r1", step_number=25, result_files=["r1"])],
            ONE: [NS(id="ftckpt_o1", step_number=25, result_files=["o1"])],
        }
        self.asked = []

    def job(self, job_id):
        self.asked.append(job_id)
        return self.jobs[job_id]

    def checkpoints(self, job_id):
        return self.cps[job_id]


def test_keeps_the_last_checkpoint_and_skips_running_jobs():
    tf = FakeTF()
    assert files_to_delete(tf, DONE) == ["f1", "f2", "f3", "f4"]
    assert files_to_delete(tf, RUNNING) is None
    assert files_to_delete(tf, ONE) == []
    # the last checkpoint goes too once its adapter is stored locally
    assert sorted(files_to_delete(tf, DONE, downloaded={"ftckpt_e3"})) == [
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
    ]
    assert files_to_delete(tf, DONE, downloaded={"ftckpt_e1"}) == [
        "f1",
        "f2",
        "f3",
        "f4",
    ]


def test_prune_deletes_once_and_leaves_running_jobs_for_later():
    tf, deleted, done, seen = FakeTF(), [], {}, []
    on_job = lambda job_id, n: seen.append((job_id, n, dict(done)))
    assert prune(tf, [DONE, RUNNING, ONE], done, deleted.append, on_job=on_job) == (
        2,
        4,
    )
    assert deleted == ["f1", "f2", "f3", "f4"]
    assert done == {DONE: 4, ONE: 0}
    # each job is reported once its files are gone and it is recorded
    assert seen == [(DONE, 4, {DONE: 4}), (ONE, 0, {DONE: 4, ONE: 0})]
    tf.asked.clear()
    assert prune(tf, [DONE, RUNNING, ONE], done, deleted.append) == (0, 0)
    assert tf.asked == [RUNNING]


def test_a_dry_run_changes_nothing():
    tf, done = FakeTF(), {}
    assert prune(tf, [DONE, ONE], done, None) == (2, 4)
    assert done == {}


def test_job_ids_and_downloaded_checkpoints_come_from_local_files(tmp_path):
    (tmp_path / "bench" / "tf").mkdir(parents=True)
    (tmp_path / "bench" / "tf" / "jobs.json").write_text(
        json.dumps({"a-s0": DONE, "_valid": "file-1"})
    )
    (tmp_path / "rows.jsonl").write_text(json.dumps({"job_id": ONE}) + "\n")
    (tmp_path / "tf").mkdir()
    (tmp_path / "tf" / "pruned.json").write_text(json.dumps({RUNNING: 1}))
    ckpt = tmp_path / "bench" / "ftckpt_e3"
    ckpt.mkdir()
    (ckpt / "adapter_model.safetensors").write_bytes(b"x")
    (ckpt / "tokenizer.json").write_text(RUNNING + " " * 2_000_001)
    assert local_job_ids(tmp_path) == {DONE, ONE}
    assert downloaded_checkpoints(tmp_path) == {"ftckpt_e3"}
