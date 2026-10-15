# B10 and B11: noise floor and the causal check

The code is in `core/interventions/stats.py`. To rerun the numbers below, use `python -m scripts.noise_floor --runs artifacts/b9/sweep-500 --metric exact_match`.

## Runs

The sweep ran locally on 2026-10-16 with `jobs/kaggle/sweep.py`:

- Qwen3-0.6B with LoRA r 16, on 500 BANKING77 training examples, for 3 epochs with a linearly decaying learning rate of 1e-4.
- Metrics come from 100 validation examples: exact match of greedy answers, and validation loss on the answer tokens.
- There are 5 configs with 3 seeds each: the healthy recipe, the learning rate ×0.1 and ×0.03, and LoRA rank 4 and 2.

| Config | Exact match | Valid loss |
|---|---|---|
| healthy | 0.420 ± 0.010 | 0.520 ± 0.011 |
| lr ×0.1 | 0.050 ± 0.000 | 1.966 ± 0.007 |
| lr ×0.03 | 0.030 ± 0.000 | 2.534 ± 0.006 |
| rank 4 | 0.417 ± 0.021 | 0.526 ± 0.008 |
| rank 2 | 0.417 ± 0.023 | 0.528 ± 0.006 |

## Noise floor and power (B10)

The noise floor is the pooled standard deviation between seeds within each config. For exact match it is 0.0146; for valid loss it is 0.0077. With a one-sided test at α 0.05 and 80% power, the smallest effect a causal check can find is:

| Seeds per arm | Exact match | Valid loss |
|---|---|---|
| 2 | 0.036 | 0.019 |
| 3 | 0.030 | 0.016 |
| 5 | 0.023 | 0.012 |

The learning-rate fault costs 0.37 of exact match, so 2 seeds per arm are enough to find it. For an effect of 0.02, `seeds_for_power` asks for 7 seeds.

## Decision rule (B11)

`causal_check(fix, control, min_effect, noise_sd)` takes one metric value per seed and arm. The fix is the targeted change and the control is a change of the same cost.

- `confirmed` needs three things: the fix beats the control by at least `min_effect`, the 95% interval of the difference lies above zero, and a one-sided permutation test gives p ≤ 0.05.
- `rejected` means the whole interval lies below `min_effect`, so the fix gains less than it has to.
- `inconclusive` covers everything else. The planner can add seeds or drop the hypothesis.

The interval uses the noise-floor SD when one is given, and otherwise the pooled SD of the runs with Student's t. With 3 seeds per arm the permutation test cannot go below p = 1/20, so a confirmation needs every fix run above every control run.

The local sweep gives a check of the rule. It compares the healthy recipe, taken as the fix, against each fault, taken as the run where the cause stays, with `min_effect` set to two noise-floor SDs:

| Fault | Exact match | Valid loss |
|---|---|---|
| lr ×0.1, ×0.03 | confirmed, +0.37 and +0.39 | confirmed |
| rank 4, rank 2 | rejected, +0.003 [−0.020, +0.027] | inconclusive, +0.006 to +0.008 |

## Limits

- The numbers cover one task, one model and one data size. B10 also asks for 200 and 1000 examples; the same sweep with `--train` from `scripts.measure_post_training --dry-run` covers those sizes.
- All runs use the same 100 validation examples, so the noise floor is training noise only. Generalizing to new examples adds sampling error, about 0.05 for exact match near 0.42 on 100 examples.
- Token Factory checkpoints report a `valid_loss` that jumps between epochs (0.43, 0.002, 0.36, 0.04 in one job) and leave `full_valid_loss` empty. Causal checks on Token Factory jobs therefore need their own evaluation of the checkpoints.
