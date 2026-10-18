# B10: power of the implemented decision rule

`bench.causal_power` simulates the complete `causal_check` rule: minimum observed
effect, interval above zero, and the permutation gate when its resolution permits
it. This differs from the one-sided normal approximation in `seeds_for_power`.
The simulation uses independent normal seed-level metrics with known equal SD
in both arms. Values are not clipped to accuracy bounds; this is a noise model,
not a simulation of individual classification outcomes.

## Saved experiment

Each cell contains 500 trials with seed 0, noise SD 0.019 and minimum effect 0.06.
The numbers are fractions of trials returning `confirmed`.

| True effect | 2 seeds/arm | 3 seeds/arm | 4 seeds/arm |
|---|---:|---:|---:|
| 0.00 | 0.000 | 0.000 | 0.000 |
| 0.03 | 0.078 | 0.034 | 0.012 |
| 0.06 | 0.502 | 0.532 | 0.450 |
| 0.10 | 0.984 | 0.990 | 0.996 |

At the threshold, a symmetric unbiased effect estimate exceeds the threshold
about half the time. More seeds narrow its distribution but do not make the
chance of exceeding its own mean approach 80%. Plan power for an anticipated
true effect above the minimum useful effect, and evaluate the complete rule.
Do not interpret the normal-approximation seed count as an 80% guarantee for
`confirmed` under the implemented rule.

The zero-effect cells measure false confirmations under the stated assumptions.
Zero out of 500 does not prove zero risk: the Wilson upper bound is about 0.76%.
The [JSON report](results/causal-power-20261018.json) includes all verdict counts,
Wilson intervals for Monte Carlo error and hashes of the simulator and statistics
code. Those intervals do not cover uncertainty in the assumed noise floor.
The same random seed reuses noise across cells; cells are not independent samples.

With two seeds per arm, the smallest exact permutation p-value is 1/6. The
existing implementation skips that gate and uses the interval and effect rules.
For three and four seeds it applies the permutation gate. The report makes
this distinction explicit.

## Permutation implementation

The test now counts possible assignments before choosing exact enumeration or
Monte Carlo sampling. Large spaces use 20,000 draws directly, without allocating
all combinations first. The sampled estimate includes the observed assignment,
`(hits + 1) / (draws + 1)`, so a finite Monte Carlo run cannot report p=0.
Exact tests keep their existing values; historical two-seed comparisons are
unchanged. Empty arms and nonfinite values are rejected.

## Reproduce

```bash
uv run python -m bench.causal_power \
  --noise-sd 0.019 --min-effect 0.06 \
  --effects 0 0.03 0.06 0.10 \
  --seeds 2 3 4 --trials 500 --random-seed 0 \
  --out artifacts/causal-power-reproduction.json
```

Use a new output file. No training or provider requests are performed. The CLI
bounds seed counts and trials to limit runtime, but larger combinations can
still be expensive. This audit does not close B10's missing empirical sweeps
across dataset sizes, tasks and models, or establish transfer of the local SD
to Token Factory.
