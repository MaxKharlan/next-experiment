# answer_truncation: damage, then fix against control

Answers cut by the user's own data preparation (fault `answer_truncation`,
`bench/faults/data.py`) were planted in BANKING77 and trained twice on 2026-10-19:
with the local LoRA lab on a Kaggle T4 (3 seeds, 15 runs), then as six Token
Factory jobs (mode A). Both show how much the fault costs and whether the targeted
fix beats a same-size control.

| | Healthy | 30% cut | Target | Control | Effect, 95% interval | Verdict |
|---|---:|---:|---:|---:|---|---|
| Kaggle, 3 seeds | 0.883 | 0.658 | 0.804 | 0.621 | +0.183, 0.173–0.194 | confirmed |
| Token Factory, 2 seeds | 0.926 | 0.681 | 0.859 | 0.662 | +0.197, 0.160–0.235 | confirmed |

Exact match on the 770 check examples. Healthy and 30% cut on Token Factory are
one job each.

Files: `docs/results/answer-cut-kaggle-20261019/` and
`docs/results/answer-cut-tf-20261019/`.

## Why this fault

Token Factory does not cut long examples. A job with an example longer than
`context_length` fails while it processes the dataset ("Sequence length is too
long: 9013 > 8192"), and a `context_length` below 8192 is refused when the job is
created (two jobs on 2026-10-18). Silent truncation of long inputs is therefore a
mode-B fault. The mode-A version is code that trims examples before the upload,
keeps their start and drops the end of the assistant answer.

## Kaggle: setup

- Model `Qwen/Qwen3-0.6B` (revision `c1899de2`), `jobs/kaggle/train_lora.py` with
  the healthy recipe: 1 epoch, batch 8, learning rate 1e-4, LoRA r 16, alpha 16,
  dropout 0.05, linear decay. Seeds 0, 1 and 2 for every arm.
- Data: the A1 splits (`bench/tasks/banking77.py`). Damaged copies cut half of
  each answer in 10% and 30% of train and validation, longest examples first;
  check and test are untouched.
- Metric: exact match of greedy answers on all 770 check examples, scored by
  `core.eval` (sha256 `5a3e4f7f…`, the same file as the local A1 build).
- Fix and control at 30%: the `answer_cut` probe flags 2308 of the 2536 cut
  training answers and nothing else. The target drops the flagged examples; the
  control drops 2308 random unflagged ones (`core/interventions/data_pairs.py`).
  Both keep 6144 examples; the control trains about 9% more tokens.
- Code: commit `2cb1661`. The notebook as run is `notebook-as-run.ipynb.json`; it embeds the
  scripts now in `jobs/kaggle/answer_cut_effect.py` and `answer_cut_pair.py`.

## Kaggle: results

Exact match on check, mean of 3 seeds (sample SD in brackets):

| Arm | Exact match | Cut answers given by the model |
|---|---:|---:|
| Healthy | 0.883 (0.011) | 0% |
| 10% of answers cut | 0.802 (0.011) | 8.4% |
| 30% of answers cut | 0.658 (0.006) | 24.1% |
| 30%, target: drop flagged | 0.804 (0.003) | |
| 30%, control: drop as many random | 0.621 (0.006) | |

- Cutting 10% of the answers costs 0.081 exact match; 30% costs 0.225. The model
  gives cut labels such as `card_a` on a slightly smaller share of check queries
  than the share of cut answers in training. Training metrics stay good: validation token accuracy
  is 0.95–0.97 in every run.
- `scripts.analyze_kaggle` on the 30% bundle with `--min-effect 0.06`, the project
  threshold chosen before the comparison: `confirmed`, effect 0.183, 95% interval
  0.173–0.194, permutation p 0.05, the smallest value 3 seeds per arm allow.
- The fix recovers most of the damage but not all: 0.804 against 0.883 healthy.
  It trains on 27% fewer examples, and 228 cut answers the probe misses stay in.
  A confirmed result supports dropping the flagged examples against the control;
  it does not show full recovery.
- The control is worse than the damaged run (0.621 against 0.658): it loses 27% of
  the clean examples and keeps the cut ones, so it pays the cost of a smaller file
  without the fix.

Limits: one task, one model, one recipe, one Kaggle environment. The noise SD
comes from these runs, not from Token Factory, so 0.06 is used as a decision
threshold only. Seed 0 repeated the earlier interactive session exactly
(0.8909, 0.8065, 0.6584).

## Kaggle: files

- `bundles/answers-cut-30.json`, `bundles/answers-cut-10.json`: runs in
  `docs/kaggle-results.schema.json` format. The 10% bundle has no target or
  control, so its analysis stops at the quality drop.
- `analysis-30.json`, `analysis-10.json`: output of `scripts.analyze_kaggle`.
- `evidence/`: the notebook's `results.json` and `pair.json`, every run's
  `summary.json` from `train_lora.py` and the fault records. The summaries score
  validation, so the bundles do not reference them by `summary_path`.
- `notebook-as-run.ipynb.json`: the notebook as it ran, byte for byte; the
  `.json` name keeps the formatter from rewriting it.

## Token Factory

Six Post-training jobs on 2026-10-19 with the same recipe, LoRA without packing
(`scripts/answer_cut_tf.py`): healthy and 30% cut once each, target and control
with seeds 0 and 1. Token Factory ignores the job seed, so each seed is its own
order of the training file, as in `core/interventions/tf_runner.py`; the two seeds
of each arm trained different weights. The last checkpoint of every job was scored
locally with `core.eval` on the same check file as on Kaggle.

| Job | Exact match | Trained tokens |
|---|---:|---:|
| Healthy | 0.926 | 481,986 |
| 30% of answers cut | 0.681 | 476,660 |
| Target s0 / s1: drop flagged | 0.858 / 0.860 | 325,019 each |
| Control s0 / s1: drop as many random | 0.661 / 0.662 | 354,394 each |

- The fault costs 0.245 exact match on Token Factory: the job succeeds and
  nothing in it warns.
- `causal_check` with the project's Token Factory noise SD 0.019 and minimum effect
  0.06: `confirmed`, effect 0.197, 95% interval 0.160–0.235. With 2 seeds the
  permutation test cannot reach 0.05 (p 0.17), so the interval decides, as the rule
  documents.
- The arms rank as on Kaggle and the effect is close (0.197 against 0.183).
  Absolute scores are higher on Token Factory (healthy 0.926 against 0.883): its
  trainer differs from the local lab, so the two are not interchangeable.
- Cost: 2,317,472 trained tokens, $0.93 at $0.40 per million; the estimate before
  submission was $0.94. Token Factory counted about 1.7% fewer tokens than the
  local chat template.

Files in `docs/results/answer-cut-tf-20261019/`: `plan.json` (files, hashes,
token estimate), `jobs.json` (job IDs, statuses, trained tokens, adapter hashes),
`results.json` (exact match per job).

## Reproduce

On Kaggle: import `jobs/kaggle/answer_cut.ipynb`, set `REVISION`, choose GPU T4
and Internet on, then Save Version > Save & Run All (about 4 hours). Download the
output, then from the repository root:

```bash
python -m jobs.kaggle.export_answer_cut <output>/next-experiment \
    --out artifacts/kaggle/answer-cut --env gpu="Tesla T4"
python -m scripts.analyze_kaggle artifacts/kaggle/answer-cut/answers-cut-30.json \
    --min-effect 0.06 --out artifacts/kaggle/answer-cut/analysis-30.json
```

On Token Factory (paid, about $0.93; needs the A1 splits and the 30% copy):

```bash
PYTHONPATH=. python scripts/answer_cut_tf.py prepare
PYTHONPATH=. python scripts/answer_cut_tf.py submit
PYTHONPATH=. python scripts/answer_cut_tf.py collect
PYTHONPATH=. python scripts/answer_cut_tf.py evaluate
```

Analysis only, from the saved bundles:

```bash
python -m scripts.analyze_kaggle \
    docs/results/answer-cut-kaggle-20261019/bundles/answers-cut-30.json --min-effect 0.06
```
