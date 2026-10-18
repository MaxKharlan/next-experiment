# B6: duration fits for three small models

All 15 Token Factory jobs succeeded on October 19, 2026. Each model used the same
nested BANKING77 training slices: 50, 200 and 1000 examples at one epoch, 200 at
three epochs, and 500 at two epochs. The last condition was selected for validation
before any measurements; the other four fit the coefficients.

Training-stage duration is `intercept + coefficient * trained_tokens / 1000 +
extra_epoch * (epochs - 1)`, in seconds.

| Base model | Intercept | Seconds / 1k tokens | Extra epoch | Validation actual | Prediction | Relative error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 97.805 | 3.189 | 12.200 | 298 s | 290.662 s | 2.46% |
| Qwen3-1.7B | 104.039 | 3.179 | 12.249 | 300 s | 296.383 s | 1.21% |
| Llama-3.2-1B-Instruct | 93.434 | 2.565 | 8.017 | 287 s | 287.768 s | 0.27% |

All fits pass the preselected 30% relative-error limit and have nonnegative
coefficients. `estimate_job` now selects their registered durations by exact model
ID. An explicit `duration=` overrides selection; unmeasured models use the historical
Llama-8B fallback. `duration_from_report` refuses a model mismatch, nonfinite or
negative coefficients, or a failed validation check.

Dataset, startup and finalization stages use medians from the four fitting runs.
The held-back condition validates only the training-stage prediction. Provider
queueing affects elapsed time: for Qwen3-0.6B, the 50-example job took 941 seconds
end to end, longer than the 1000-example job's 896 seconds despite having a shorter
training stage. These fits therefore do not guarantee completion time, isolate
GPU compute time, or validate prediction on other tasks or hardware allocations.
There is only one run per condition.

The 15 jobs report an estimated total cost of $0.2112952 at the recorded short-context
rate of $0.40 per million trained tokens. No billing receipt was available for
reconciling these individual jobs. Dollar estimates remain token-based; fitting
time coefficients does not change the price table.

The [protocol](results/b467-20261019/protocol.md) records the budget, hashes and
parameters. [cost-fit.json](results/b467-20261019/cost-fit.json) stores unrounded
coefficients, validation job IDs and fit errors. All original summaries and the
plan are in `results/b467-20261019/costs/`.

Refit from the saved measurements without API calls:

```bash
uv run python -m bench.cost_fit docs/results/b467-20261019/costs \
  --out artifacts/model-cost-refit.json
```

The original [Llama-8B measurements](b6-post-training.md) remain the source of the
fallback model. This batch does not fill B5's missing A2/A3 reference runs.
