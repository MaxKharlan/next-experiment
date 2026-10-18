# Temperature scaling for candidate confidence

`bench.temperature` extends the [confidence audit](confidence-calibration.md)
with a fitted binary temperature. It calibrates the confidence that the selected
candidate is correct. It does not calibrate the full fault distribution or the
probability that a causal verdict is correct, and it changes no live agent behavior.

For reported confidence `p`, the adjusted confidence is
`sigmoid(logit(p) / T)`. Probabilities are clipped to `[1e-6, 1 - 1e-6]` before
converting to logits. The temperature is fitted separately for each method by
minimizing binary negative log likelihood on dev, within `[0.05, 20]`. Both correct
and incorrect candidates must occur in dev. The optimizer uses the monotonic
derivative in inverse temperature; it also checks the bounds and the identity
transform. The model records whether the optimum is at a search boundary.

Generate a dev audit, then fit a temperature:

```bash
uv run python -m bench.calibration \
  docs/results/dev-routing-20261018/rows.json \
  --split dev --out artifacts/calibration/routing-dev.json

uv run python -m bench.temperature fit \
  artifacts/calibration/routing-dev.json \
  --out artifacts/calibration/routing-temperature.json
```

When matching heldout predictions have been audited, evaluate the saved model:

```bash
uv run python -m bench.temperature evaluate \
  artifacts/calibration/routing-heldout.json \
  --model artifacts/calibration/routing-temperature.json \
  --out artifacts/calibration/routing-heldout-adjusted.json
```

The heldout audit must be generated from actual heldout predictions using
`bench.calibration --split heldout`. The evaluation step requires the same method
names and case IDs disjoint from dev. It never fits a parameter. Reports include
before/after ECE, binary Brier, NLL and adjusted confidence for each case. Source
hashes bind the fitted model to its audit and the evaluation to both inputs.
Existing files are never overwritten. Split labels describe the supplied audit;
they cannot prove that a human has never inspected those cases.

## Dev fit from the routing experiment

Both methods have identical input probabilities and yield `T = 1.820735` on the
five dev cases. Dev NLL changes from `0.640790` to `0.591295`. This is the fitting
objective measured on the fitting data, not evidence of improved generalization.
No heldout evaluation was run. Five dev cases provide a weak basis for fitting;
keep this model experimental. The historical heldout cases have already been
inspected, so a follow-up evaluation on them must be described as exploratory.

This adds the fitting tool for B12. Validation on disjoint data and any decision
to use calibrated confidence in the live product remain separate work.
