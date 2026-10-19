"""Offline CPU quality experiment for local code repairs on synthetic sequences.

No pretrained weights, external datasets or services are used. A small GRU is
trained from scratch. This measures this fixture, not LLM or provider transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from bench.faults.labels import (
    include_prompt_labels,
    mask_padding_labels,
    shift_labels_once,
)
from core.agent.context import InvestigationContext
from core.agent.label_checks import check_training_labels
from core.agent.schemas import Hypothesis
from core.interventions.code_faults import (
    CODE_FAULTS,
    LocalCodePairRunner,
    make_code_pair,
)
from core.interventions.stats import causal_check
from core.training.parameters import audit_trainable_parameters
from jobs.kaggle.train_lora import backward_microbatches, collate

EOS = 2


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def corpus(seed, count):
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        category = i % 3
        prefix = [rng.randrange(9, 25) for _ in range(rng.randrange(4, 13))]
        prompt = [1, *prefix, 3 + category]
        # Different answer lengths expose the microbatch weighting distinction.
        answer = [6 + category] * (category + 1) + [EOS]
        rows.append(
            {
                "input_ids": prompt + answer,
                "labels": [-100] * len(prompt) + answer,
                "prompt": prompt,
                "answer": answer,
            }
        )
    return rows


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(25, 12)
        self.rnn = nn.GRU(12, 24, batch_first=True)
        self.head = nn.Linear(24, 25)

    def forward(self, input_ids, attention_mask=None):
        hidden, _ = self.rnn(self.embedding(input_ids))
        return SimpleNamespace(logits=self.head(hidden))


@torch.no_grad()
def predict(model, rows):
    model.eval()
    predictions = []
    for row in rows:
        sequence = list(row["prompt"])
        answer = []
        for _ in range(6):
            token = int(model(torch.tensor([sequence])).logits[0, -1].argmax())
            answer.append(token)
            sequence.append(token)
            if token == EOS:
                break
        predictions.append(
            {
                "expected": row["answer"],
                "generated": answer,
                "exact_match": answer == row["answer"],
                "terminated": answer[-1] == EOS,
            }
        )
    return predictions


def execute_training(train_rows, test_rows, hp, fault, seed, directory):
    torch.manual_seed(seed)
    model = TinyDecoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"])
    rows = [{"input_ids": r["input_ids"], "labels": r["labels"]} for r in train_rows]
    observed = {
        "prompt_loss": include_prompt_labels,
        "eos_as_pad": lambda r: mask_padding_labels(r, EOS),
        "double_shift": shift_labels_once,
    }.get(fault, lambda r: r)(rows)
    audit = check_training_labels(
        {"reference": rows, "observed": observed, "eos_token_id": EOS}
    )
    rng = random.Random(seed)
    history = []
    for step in range(hp["steps"]):
        chosen = rng.sample(observed, hp["gradient_accumulation_steps"])
        batches = [collate([row], 0, "cpu") for row in chosen]
        optimizer.zero_grad()
        loss, _, targets = backward_microbatches(
            model, batches, microbatch_mean=fault == "microbatch_mean"
        )
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"step": step + 1, "loss": loss, "targets": targets})
    predictions = predict(model, test_rows)
    metric = sum(p["exact_match"] for p in predictions) / len(predictions)
    torch.save(model.state_dict(), directory / "weights.pt")
    write_json(
        directory / "run.json",
        {
            "seed": seed,
            "code_fault": fault,
            "hyperparameters": hp,
            "exact_match": metric,
            "termination_rate": sum(p["terminated"] for p in predictions)
            / len(predictions),
            "parameter_audit": audit_trainable_parameters(model),
            "label_audit": audit,
            "loss_curve": history,
            "predictions": predictions,
        },
    )
    return metric


def run(out, seeds, steps=100):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Require at least two distinct seeds")
    if any(type(seed) is not int or seed < 0 for seed in seeds) or steps < 1:
        raise ValueError("Require nonnegative seeds and positive steps")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train_rows, test_rows = corpus(1729, 96), corpus(2718, 48)
    assert not (
        {tuple(r["input_ids"]) for r in train_rows}
        & {tuple(r["input_ids"]) for r in test_rows}
    )
    write_json(out / "train.json", train_rows)
    write_json(out / "test.json", test_rows)
    hp = {"learning_rate": 0.015, "steps": steps, "gradient_accumulation_steps": 4}
    protocol = {
        "scope": "Synthetic CPU GRU; no pretrained model or provider transfer",
        "seeds": seeds,
        "hyperparameters": hp,
        "test_used_for_selection": False,
        "metric": "greedy sequence exact match including EOS; maximum 6 new tokens",
        "min_effect": 0.1,
        "service_cost_usd": 0.0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "datasets": {
            name: hashlib.sha256((out / f"{name}.json").read_bytes()).hexdigest()
            for name in ("train", "test")
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(out / "protocol.json", protocol)
    results = []
    # Frozen protocol: report every fault, including failures to show an effect.
    for fault_class, fault in [*CODE_FAULTS.items(), ("pad_eos_masking", None)]:
        name = fault or "healthy_negative_control"
        ctx = InvestigationContext(
            "offline", "B", "synthetic", {"code_fault": fault, "hyperparameters": hp}
        )
        hypothesis = Hypothesis(
            id="h1",
            fault_class=fault_class,
            prior=0.8,
            location="trainer",
            rationale="Injected fixture",
        )
        target, control = make_code_pair(hypothesis, ctx)
        runner = LocalCodePairRunner(
            hp,
            seeds,
            out / name,
            lambda hp, fault, seed, directory: execute_training(
                train_rows, test_rows, hp, fault, seed, directory
            ),
        )
        runner(target, control)
        result = causal_check(
            target.metrics,
            control.metrics,
            min_effect=protocol["min_effect"],
            noise_sd=None,
        )
        record = {
            "fault": name,
            "target": asdict(target),
            "control": asdict(control),
            "result": asdict(result),
        }
        results.append(record)
        write_json(out / "results.json", results)
        print(name, result.status, target.metrics, control.metrics, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    run(args.out, args.seeds, args.steps)


if __name__ == "__main__":
    main()
