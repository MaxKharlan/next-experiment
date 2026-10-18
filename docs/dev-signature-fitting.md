# B16: fitting probe signatures on dev

`bench.fit_signatures` runs the four recipe probes on every dev case, independently
of the agent's chosen trajectory. It uses hyperparameters, checkpoint losses and
truth labels. It does not use held-out metrics, task text, model calls, training
files or cached agent diagnoses. The CLI reads only the `dev` subdirectory and
rejects cases marked with another split, duplicate IDs, duplicate provider jobs,
compound truth labels and invalid probe inputs.

For each observed single fault and probe outcome, the estimate is
`(count + alpha) / (n + alpha * number_of_outcomes)`, with `alpha=1` by default.
This symmetric Dirichlet smoothing keeps unobserved outcomes possible. An absent
fault retains its baseline probabilities. Healthy observations are reported
separately: they cannot estimate the distribution of every unspecified cause,
so `default` rows remain unchanged.

## Saved fit

The [report](results/signatures-dev-20261018/report.json) records five cases:
three `lr_too_low` and two healthy. It includes observations, counts, smoothing,
and hashes of the case files, baseline, probe source and fitted YAML.

| Probe | Fitted probabilities given `lr_too_low` |
|---|---|
| lr_range | low 0.8; ok 0.2 |
| epoch_count | high 0.2; ok 0.8 |
| lora_rank | low 0.2; ok 0.8 |
| loss_curve | barely_moved 1/6; overfit 1/6; normal 2/3 |

These are fitted frequencies on three fault cases, not an independent quality
measurement. No conclusion about improved diagnosis or calibration follows from
this fit. Other fault classes need additional dev observations.

The [YAML](results/signatures-dev-20261018/signatures.yaml) is experimental. It
retains baseline rows for unmeasured probes, including the existing B25 weight
row whose documented counts included held-out cases. Only the four fitted
recipe-probe rows have dev-only provenance; the entire YAML must not be described
as a clean dev-only model. A new independent evaluation is needed before rollout.

## Reproduce and use explicitly

```bash
uv run python -m bench.fit_signatures \
  --alpha 1 \
  --out artifacts/signatures-dev
```

The output directory must be new. No provider calls are made. To evaluate planner
operations with the frozen table, use the scoped override:

```python
from pathlib import Path
from core.planner.outcome_model import using_signatures
from core.planner.eig import expected_gain

with using_signatures(Path("artifacts/signatures-dev/signatures.yaml")):
    gain = expected_gain(
        {"h1": 0.5, "other": 0.5},
        {"h1": "lr_too_low", "other": "unlisted"},
        "lr_range",
    )
```

The override also applies to Bayes updates and probe selection, and restores the
previous table after exceptions. It is context-local; wrap the investigation in
the worker that executes it. Tables are cached by path, so treat artifacts as
immutable. The API default remains the original table pending a matched comparison.

B16 now has a reproducible fitting tool and a first fitted artifact. Broader fault
coverage, untouched evaluation and a decision about the live table remain open.
