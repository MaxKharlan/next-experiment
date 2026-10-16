"""B13 smoke test: Nemotron proposes hypotheses for a fault-6 case and writes a verdict.

The case and the causal check come from local runs of Qwen3-0.6B on 200 BANKING77
examples with a learning rate of 1e-5 (docs/b10-noise-floor.md and the B8 pair
check). Calls Super once for hypotheses and Ultra once for the verdict, about one
cent in total, and prints both with the call log totals.
"""

from __future__ import annotations

import json

from core.agent.context import InvestigationContext
from core.agent.router import Router
from core.agent.steps import propose_hypotheses, write_verdict
from core.connectors.token_factory import TokenFactory
from core.interventions.pairs import make_pair
from core.interventions.stats import causal_check
from core.telemetry.calls import CallLog

JOB_HP = {
    "n_epochs": 3,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "warmup_ratio": 0.0,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "context_length": 8192,
}
CASE = {
    "symptom": (
        "After fine-tuning the model still answers like the base model: it writes "
        "free-form descriptions instead of the intent labels from the training data. "
        "Exact match on 50 held-out queries is 2%."
    ),
    "task": "classify banking queries into 77 intent labels, answer with the label",
    "job": {
        "model": "Qwen/Qwen3-0.6B",
        "status": "succeeded",
        "train_examples": 200,
        "validation_examples": 50,
        "hyperparameters": JOB_HP,
    },
    "checkpoints": [
        {"epoch": 1, "train_loss": 3.25, "valid_loss": 2.79},
        {"epoch": 2, "train_loss": 2.66, "valid_loss": 2.51},
        {"epoch": 3, "train_loss": 2.46, "valid_loss": 2.45},
    ],
    "findings": [
        "valid_loss of the base model before training: 3.82",
        "examples are 45 to 103 tokens long, far below context_length",
        "no duplicates between train and validation",
    ],
}


def main() -> None:
    log = CallLog("smoke")
    router = Router(TokenFactory.from_settings(), log)

    hypotheses = propose_hypotheses(router, CASE)
    print("Hypotheses (Super):")
    for h in hypotheses:
        print(
            f"  {h.id} {h.fault_class} p={h.prior:.2f} at {h.location}: {h.rationale}"
        )

    lead = next((h for h in hypotheses if h.fault_class == "lr_too_low"), None)
    if lead is None:
        print("lr_too_low is not among the hypotheses, so no verdict this time")
    else:
        ctx = InvestigationContext("smoke", "A", "local", {"hyperparameters": JOB_HP})
        target, control = make_pair(lead, ctx)
        target.metrics, control.metrics = [0.16, 0.20], [0.02, 0.02]  # B8 pair runs
        result = causal_check(
            target.metrics, control.metrics, min_effect=0.03, noise_sd=0.0146
        )
        verdict = write_verdict(
            router, lead, target, control, result, JOB_HP, ["pair-lr-1"]
        )
        print("Verdict (Ultra):")
        print(json.dumps(verdict.model_dump(), indent=2))

    print("Calls:")
    for model, row in log.totals("model").items():
        print(
            f"  {model}: {row['calls']} calls, {row['tokens_in']} in, "
            f"{row['tokens_out']} out, ${row['cost_usd']:.4f}, {row['latency_ms']} ms"
        )


if __name__ == "__main__":
    main()
