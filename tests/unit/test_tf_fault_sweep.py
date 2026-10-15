from types import SimpleNamespace as NS

import pytest

from bench.faults.recipe import HEALTHY_HP
from scripts.tf_fault_sweep import plan_runs, run_grid, suffix

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"


class FakeTF:
    """Every job succeeds on its second poll and leaves one checkpoint per epoch."""

    def __init__(self):
        self.hp, self.polls, self.uploads, self.max_running = {}, {}, 0, 0

    def upload(self, path):
        self.uploads += 1
        return f"file-{self.uploads}"

    def create_job(self, model, training_file, *, validation_file, **kw):
        job_id = f"ftjob-{len(self.hp)}"
        self.hp[job_id], self.polls[job_id] = kw["hyperparameters"], 0
        running = sum(1 for n in self.polls.values() if n < 2)
        self.max_running = max(self.max_running, running)
        return job_id

    def job(self, job_id):
        self.polls[job_id] += 1
        done = self.polls[job_id] >= 2
        return NS(
            id=job_id,
            status="succeeded" if done else "running",
            created_at=100,
            finished_at=400 if done else None,
            trained_tokens=1000 if done else 0,
            trained_steps=0,
            total_steps=None,
            error=None,
        )

    def events(self, job_id):
        return [
            NS(message="Dataset 'training' processed successfully", created_at=150),
            NS(message="Dataset 'validation' processed successfully", created_at=160),
            NS(message="Training started", created_at=250),
            NS(message="Training completed successfully", created_at=330),
        ]

    def checkpoints(self, job_id):
        epochs = self.hp[job_id]["n_epochs"]
        return [
            NS(
                step_number=25 * (e + 1),
                metrics=NS(train_loss=1 / (e + 1), valid_loss=e),
            )
            for e in reversed(range(epochs))
        ]


def test_plan_and_suffix():
    runs = plan_runs(LLAMA, HEALTHY_HP, [0], ["too_many_epochs"])
    assert [r["hyperparameters"]["n_epochs"] for r in runs] == [1, 3, 6]
    assert [suffix(r) for r in runs] == [
        "b9-healthy-s0",
        "b9-ep-mild-s0",
        "b9-ep-strong-s0",
    ]


def test_run_grid_keeps_the_parallel_limit_and_resumes(tmp_path):
    runs = plan_runs(LLAMA, HEALTHY_HP, [0], None)
    tf = FakeTF()
    results = run_grid(tf, runs, tmp_path, "t", "v", parallel=2, sleep=lambda s: None)

    assert len(results) == 7
    assert tf.uploads == 2
    assert tf.max_running == 2
    assert tf.hp["ftjob-0"]["lora"] is True
    six = next(r for r in results if r["run"] == "too_many_epochs-strong-s0")
    assert six["valid_loss"] == [0, 1, 2, 3, 4, 5]
    assert six["seconds"] == {
        "datasets": 60,
        "training_started": 150,
        "training_completed": 230,
        "finished": 300,
    }
    assert six["cost_usd"] == pytest.approx(0.0004)
    assert (tmp_path / "results.jsonl").read_text().count("\n") == 7

    again = FakeTF()
    run_grid(again, runs, tmp_path, "t", "v", sleep=lambda s: None)
    assert again.hp == {}
    assert again.uploads == 0
