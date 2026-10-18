# B25: what the adapter's weights say

The probe is `lora_weights` in `core/probes/level0/weights.py`. It reads the failed job's LoRA adapter and needs no inference and no model call. To rerun the numbers below, use `uv run --group eval python -m scripts.calibrate_lora_weights --arms`; the rows go to `artifacts/b25/weights.jsonl`.

## What the probe measures

Token Factory puts a PEFT adapter in `result_files`, not merged weights (`docs/b6-post-training.md`). Each adapted matrix W changes by dW = s · B · A, where s = alpha / r. For each of the 196 adapted matrices of Qwen3-0.6B the probe computes:

- the norm of dW divided by the norm of W, from the base model's weights;
- the singular values of dW, from the r × r core of the QR factors of B and Aᵀ;
- their effective rank (Roy and Vetterli, 2007), the exponential of the entropy of the normalised singular values, divided by r.

The outcome is `small` when the median relative norm over the matrices is below 0.0195, and `normal` otherwise. The effective rank is reported with it but does not decide the outcome; the section on rank explains why.

## Data

Three sets of adapters, each labelled by whether the learning rate was too low (fault 6):

| Set | Adapters | What they are |
|---|---|---|
| B9 on a MacBook | 15 | local runs on 500 examples: healthy, and faults 6 and 8 at two strengths, with three seeds each |
| Token Factory cases | 13 | the failed jobs behind the benchmark cases, 4 dev and 9 held-out, including the healthy default recipe that `make_case` dropped |
| Token Factory arms | 86 | the arms of the benchmark's causal checks; each recipe is the case's plus the check's change |

Identical weights count once. Token Factory training is deterministic, so a control arm that repeats a recipe on the same files trains a copy, and so does a dropped case that shares its job with a kept one.

## The failed jobs

| Split | Case | Learning rate | Epochs | Rank | Median relative norm | Effective rank / r |
|---|---|---|---|---|---|---|
| held-out | ho-def-fault6-strong-s0 | 3e-6 | 3 | 8 | 0.0017 | 0.86 |
| held-out | ho-e6-fault6-strong-s0 | 3e-6 | 6 | 16 | 0.0021 | 0.80 |
| dev | ref-fault6-strong-s0 | 6e-6 | 6 | 16 | 0.0033 | 0.82 |
| held-out | ho-def-fault6-mild-s0 | 1e-5 | 3 | 8 | 0.0037 | 0.88 |
| held-out | ho-e12-fault6-strong-s0 | 6e-6 | 12 | 16 | 0.0044 | 0.83 |
| held-out | ho-e6-fault6-mild-s0 | 1e-5 | 6 | 16 | 0.0045 | 0.83 |
| dev | ref-fault6-mild-s0 | 2e-5 | 6 | 16 | 0.0073 | 0.84 |
| held-out | ho-e12-fault6-mild-s0 | 2e-5 | 12 | 16 | 0.0094 | 0.86 |
| held-out | ho-def-healthy-s0 | 1e-4 | 3 | 8 | 0.0217 | 0.91 |
| held-out | ho-e6-healthy-s0 | 1e-4 | 6 | 16 | 0.0272 | 0.89 |
| dev | ref-healthy-s0 | 2e-4 | 6 | 16 | 0.0522 | 0.90 |
| held-out | ho-e12-healthy-s0 | 2e-4 | 12 | 16 | 0.0674 | 0.92 |
| dev | ref-epochs18-s0 | 2e-4 | 18 | 16 | 0.0759 | 0.92 |

All 8 jobs with a low learning rate sit below 0.0095 and all 5 others above 0.021. The norm grows with the learning rate and with the number of epochs, less than in proportion: on the 6-epoch recipe, a learning rate ten times lower gives a norm six times smaller (0.0045 against 0.0272).

## Threshold

The threshold comes from the 4 dev cases only: halfway on a log scale between the largest fault-6 norm (0.0073) and the smallest other (0.0522), which is 0.0195. Then it was checked on data it had not seen:

- held-out cases: 9 of 9 on the right side;
- arms: 84 of 86. The 54 arms with a low learning rate span 0.0009 to 0.0096, the 32 others 0.0175 to 0.1144.

The two arms on the wrong side are the epochs check on ho-e6-healthy: the healthy recipe trained for 2 epochs instead of 6, with norms 0.0175 and 0.0178. They did undertrain, at 0.355 and 0.32 exact match against 0.53 and 0.51 for the 6-epoch control. So `small` means that training did not move the weights enough, whatever the reason. In the fault catalog that is evidence for lr_too_low, but a job with too few epochs looks the same, and the catalog has no class for it.

## The threshold does not carry over from the MacBook runs

| B9 run on the MacBook | Median relative norm | Effective rank / r |
|---|---|---|
| healthy, learning rate 1e-4, rank 16 | 0.0030 | 0.72 |
| learning rate 1e-5 | 0.0007 | 0.72 |
| learning rate 3e-6 | 0.0003 | 0.72 |
| rank 4 | 0.0057 | 0.88 |
| rank 2 | 0.0081 | 0.95 |

The order is the same as on Token Factory, with fault 6 several times below healthy, but the whole scale is about ten times smaller: a healthy run at 1e-4 reaches 0.0030 here in 189 steps, against 0.0272 on Token Factory in 150. With the Token Factory threshold, only the 6 fault-6 runs of the 15 land on the right side. The project document planned to set the thresholds on B9 and check them on Token Factory; because of this gap they were set on Token Factory jobs, and B9 only confirms the direction. A threshold has to come from jobs trained the way the judged job was trained.

## Rank

On B9 the effective rank does what the project document expected of fault 8: a rank-16 adapter uses 0.72 of its rank, rank 4 uses 0.88 and rank 2 uses 0.95. It does not turn into a usable outcome, for two reasons:

- On Token Factory, healthy rank-16 adapters already use 0.78 to 0.93 of their rank and rank-8 adapters 0.84 to 0.91, so a high ratio does not single out a small rank. There are no Token Factory jobs with rank 4 or less to calibrate on.
- A smaller rank did not hurt this task: on B9, rank 4 and rank 2 scored 0.417 exact match, like the healthy 0.420 (`docs/b10-noise-floor.md`). A rank signal would raise alarms on jobs that work.

So the probe reports the effective rank on its card and the signature table has no fault-8 row for it.

## In the planner

`core/probes/signatures.yaml` has a `lora_weights` row from the counts on the 13 failed jobs, with one added to each outcome: P(small | lr_too_low) = 0.90 and P(small) = 0.14 for every other class. The planner treats probes as independent given the cause, so when it runs both `lr_range` and `lora_weights` it counts the learning rate twice. The weights probe measures what training did; `lr_range` reads the config.

A probe can now say it is not available: `lora_weights` runs only when `ctx.files["adapter"]` points at a local adapter, and the agent's loop skips probes that are not available. `scripts/investigate_tf.py --job <failed job ID>` downloads the job's last checkpoint and turns the probe on; `--adapter <dir>` uses an adapter already on disk. The API's Token Factory runner and the benchmark can read cached measurements prepared by `scripts.prepare_weight_probes`. Enable them with `NEXT_EXPERIMENT_WEIGHTS=1`. The cache is checked against the job ID, model and recipe. Next Experiment, ReAct and the checklist receive the same weight probe. Historical benchmark results exclude this input.

The first run for a model reads the base model's safetensors once and caches the norms in `artifacts/probes/`. After that one adapter takes 0.1 to 0.5 s on a MacBook.
