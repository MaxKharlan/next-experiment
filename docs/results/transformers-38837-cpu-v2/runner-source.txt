"""Offline CPU mechanism reproduction of Transformers issue #38837.

Uses an isolated historical Trainer stack and a new, tiny linear model.
No model downloads, provider APIs or previously trained adapters are used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import tempfile
from pathlib import Path

CASES = ((8, 2), (10, 2), (10, 3))
RATES = (0.0, 0.05)
VERSIONS = {"torch": "2.6.0", "transformers": "4.51.3", "accelerate": "1.6.0"}
TOLERANCE = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as file:
        file.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(root):
    from accelerate import Accelerator
    from transformers import Trainer

    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise ValueError(f"Use the isolated historical runtime: {VERSIONS}")
    root.mkdir(parents=True, exist_ok=False)
    save(
        root / "protocol.json",
        {
            "id": "transformers-38837-cpu-mechanism-v2",
            "source_url": "https://github.com/huggingface/transformers/issues/38837",
            "upstream_version": "4.51.3",
            "versions": versions,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "script_sha256": sha(__file__),
            "trainer_sha256": sha(inspect.getfile(Trainer)),
            "accelerator_sha256": sha(inspect.getfile(Accelerator)),
            "cases": [{"examples": n, "accumulation": k} for n, k in CASES],
            "learning_rates": list(RATES),
            "microbatch_size": 2,
            "epochs": 1,
            "max_steps": "ceil(examples / microbatch_size / accumulation); forces one complete pass despite historical Trainer floor rounding",
            "preflight": "v1 stopped at an update-count mismatch because default max_steps omitted remainder groups. Cases, rates and tolerance are unchanged.",
            "seed": 17031,
            "model": "Float64 linear regression, weight [0.25, 0.5], bias 0.125",
            "data": "Every row x=[1,2], y=1. Synthetic, created by this script.",
            "optimizer": "SGD, constant learning rate, no momentum/weight decay/clipping",
            "arms": [
                "Unmodified transformers 4.51.3 Trainer",
                "Same Trainer with division by actual microbatches in the group",
                "Independent analytic full-batch SGD reference",
            ],
            "expected": "Complete groups agree. Original final remainder gradient scales by remainder/K. Corrected gradients and parameters agree with the analytic reference.",
            "absolute_tolerance": TOLERANCE,
            "stopping": "Any mismatch fails the mechanism reproduction. No parameter search or pretrained-quality claim.",
            "scope": "Adapted CPU mechanism reproduction; not the reporter's BERT/GPU run, not a downstream quality experiment, not evidence about current Token Factory.",
            "service_spend_usd": 0,
        },
    )
    (root / "protocol.sha256").write_text(sha(root / "protocol.json") + "\n")


def analytic(n, accumulation, rate, *, buggy=False):
    parameters = [0.25, 0.5, 0.125]
    history = []
    microbatches = n // 2
    for start in range(0, microbatches, accumulation):
        count = min(accumulation, microbatches - start)
        error = parameters[0] + 2 * parameters[1] + parameters[2] - 1
        factor = count / accumulation if buggy else 1
        gradient = [2 * error * x * factor for x in (1, 2, 1)]
        parameters = [p - rate * g for p, g in zip(parameters, gradient)]
        history.append(
            {
                "gradient": gradient,
                "parameters": list(parameters),
                "microbatches": count,
            }
        )
    return history


def run_trainer(n, accumulation, rate, fixed):
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 1, dtype=torch.float64)
            with torch.no_grad():
                self.layer.weight.copy_(torch.tensor([[0.25, 0.5]]))
                self.layer.bias.fill_(0.125)

        def forward(self, input_ids, labels):
            values = self.layer(input_ids.double()).flatten()
            return {"loss": ((values - labels.double()) ** 2).mean(), "logits": values}

    class ActualGroupTrainer(Trainer):
        def get_batch_samples(self, epoch_iterator, num_batches, device):
            batch_samples, count = super().get_batch_samples(
                epoch_iterator, num_batches, device
            )
            self.actual_group_size = len(batch_samples)
            return batch_samples, count

        def training_step(self, model, inputs, num_items_in_batch=None):
            configured = self.args.gradient_accumulation_steps
            try:
                self.args.gradient_accumulation_steps = self.actual_group_size
                return super().training_step(model, inputs, num_items_in_batch)
            finally:
                self.args.gradient_accumulation_steps = configured

    class Capture(TrainerCallback):
        def __init__(self):
            self.rows = []

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            self.rows.append(
                {
                    "gradient": torch.cat(
                        [p.grad.flatten() for p in model.parameters()]
                    ).tolist()
                }
            )

        def on_step_end(self, args, state, control, model=None, **kwargs):
            self.rows[-1]["parameters"] = torch.cat(
                [p.detach().flatten() for p in model.parameters()]
            ).tolist()

    torch.set_num_threads(1)
    callback = Capture()
    with tempfile.TemporaryDirectory() as directory:
        args = TrainingArguments(
            output_dir=directory,
            use_cpu=True,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=accumulation,
            num_train_epochs=1,
            max_steps=math.ceil(n / 2 / accumulation),
            learning_rate=rate,
            lr_scheduler_type="constant",
            optim="sgd",
            weight_decay=0,
            max_grad_norm=0,
            seed=17031,
            data_seed=17031,
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            dataloader_pin_memory=False,
        )
        trainer = (ActualGroupTrainer if fixed else Trainer)(
            model=Linear(),
            args=args,
            train_dataset=[{"input_ids": [1.0, 2.0], "labels": 1.0} for _ in range(n)],
            callbacks=[callback],
        )
        if trainer.model_accepts_loss_kwargs:
            raise ValueError("This experiment requires the mean-loss branch")
        trainer.train()
    return callback.rows


def validate_result(result):
    expected_cases = [(n, k, rate) for n, k in CASES for rate in RATES]
    if len(result["cases"]) != len(expected_cases):
        raise ValueError("Missing cases")
    for row, (n, k, rate) in zip(result["cases"], expected_cases):
        if (row["examples"], row["accumulation"], row["learning_rate"]) != (n, k, rate):
            raise ValueError("Changed design")
        for arm in ("original", "corrected"):
            expected = analytic(n, k, rate, buggy=arm == "original")
            observed = row[arm]
            if len(observed) != len(expected):
                raise ValueError("Optimizer update count differs")
            for actual, reference in zip(observed, expected):
                for key in ("gradient", "parameters"):
                    if len(actual[key]) != 3 or any(
                        isinstance(a, bool)
                        or not isinstance(a, (int, float))
                        or not math.isfinite(a)
                        or not math.isclose(a, b, rel_tol=0, abs_tol=TOLERANCE)
                        for a, b in zip(actual[key], reference[key])
                    ):
                        raise ValueError(f"{arm} {key} differs from analytic reference")
    if (
        result["status"] != "mechanism_reproduced"
        or result["downstream_quality_measured"]
    ):
        raise ValueError("Unsupported conclusion")


def execute(root):
    from accelerate import Accelerator
    from transformers import Trainer

    protocol = json.loads((root / "protocol.json").read_text())
    digest = sha(root / "protocol.json")
    if (root / "protocol.sha256").read_text().strip() != digest:
        raise ValueError("Protocol changed")
    for key, path in (
        ("script_sha256", __file__),
        ("trainer_sha256", inspect.getfile(Trainer)),
        ("accelerator_sha256", inspect.getfile(Accelerator)),
    ):
        if sha(path) != protocol[key]:
            raise ValueError(f"Frozen source changed: {key}")
    if {name: importlib.metadata.version(name) for name in VERSIONS} != protocol[
        "versions"
    ]:
        raise ValueError("Runtime changed")
    if any((root / name).exists() for name in ("results.json", "failed-results.json")):
        raise FileExistsError("Results already exist; verify instead of rerunning")
    result = {
        "protocol_sha256": digest,
        "status": "mechanism_reproduced",
        "downstream_quality_measured": False,
        "cases": [
            {
                "examples": n,
                "accumulation": k,
                "learning_rate": rate,
                "original": run_trainer(n, k, rate, False),
                "corrected": run_trainer(n, k, rate, True),
            }
            for n, k in CASES
            for rate in RATES
        ],
    }
    try:
        validate_result(result)
    except ValueError as error:
        save(
            root / "failed-results.json",
            {**result, "status": "reproduction_not_achieved", "error": str(error)},
        )
        raise
    save(root / "results.json", result)
    save(
        root / "manifest.json",
        {
            name: sha(root / name)
            for name in ("protocol.json", "protocol.sha256", "results.json")
        },
    )


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if set(manifest) != {"protocol.json", "protocol.sha256", "results.json"}:
        raise ValueError("Manifest coverage mismatch")
    for name, expected in manifest.items():
        if sha(root / name) != expected:
            raise ValueError("Evidence drift")
    result = json.loads((root / "results.json").read_text())
    if result["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Result binding mismatch")
    validate_result(result)
    return {
        "status": result["status"],
        "cases": len(result["cases"]),
        "trainer_runs": 12,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out)
    elif args.action == "run":
        execute(args.out)
        print(json.dumps(verify(args.out)))
    else:
        print(json.dumps(verify(args.out)))


if __name__ == "__main__":
    main()
