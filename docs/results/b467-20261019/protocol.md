# B4, B6 and B7 experiment protocol

Base revision: `618db22` (PR #78), October 19, 2026. B5 is deferred because
approved A2/A3 data is absent. These experiments use existing BANKING77 data.
No Kaggle runs are part of this batch.

## Budget

The authorized total is $10, including generation and evaluation. The B6
15-job plan reserves $6.884211 at a conservative $1 per million tokens, using
serialized UTF-8 bytes plus 512 template tokens per example as the token bound.
The nine B7 jobs reserve $2.594520 at $1 per million tokens: the existing Qwen
chat tokenizer counts 28,828 tokens per epoch, and the jobs total 90 epochs.
These are local accounting reservations, not provider-enforced spending limits.

Saved B4 generation calls total $0.0012888. A failed three-attempt syntax check
lost its local usage record; $0.10 is reserved for it. Initial reservations were
$9.5800198, leaving $0.4199802 before completed training reservations were reconciled. No further training is authorized within this
protocol. Adapter exact-match evaluation runs locally and has no API inference
charge. Training reports multiply `trained_tokens` by the recorded short-context
rate of $0.40/M; they are estimates, not billing receipts.

## B6 conditions

`python -m scripts.measure_model_costs --out artifacts/b467-20261019/costs --budget 9`
creates the plan; `--execute` submits paid jobs and resumes stored IDs. The sources
are the existing 1000-example BANKING77 train set and 50-example validation set
under `artifacts/b6-llama-8b-n1000/`. The saved plan hashes every training slice
and the validation file. An unknown submission outcome stops resumption until
it is reconciled against provider jobs.

Models: Qwen3-0.6B, Qwen3-1.7B and Llama-3.2-1B-Instruct. Each uses 50, 200 and
1000 examples at one epoch, 200 at three epochs, and 500 at two epochs. Four
conditions fit `intercept + trained_tokens/1000 + epochs-1`; 500/two is selected
in advance for validation. There is one run per condition, so this does not
estimate variation between repeated submissions. Dataset, startup and finalization
stages use fitting-condition medians.

`bench.cost_fit` preserves all coefficients and validation errors, including
unusable fits. `duration_from_report` rejects model mismatches, negative or
nonfinite coefficients, and validation relative errors above 30%. Accepted
results are registered per model in the planner and can also be passed explicitly
to `estimate_job(..., duration=...)`. The historical Llama-8B estimate is the fallback
for models without accepted measurements. Provider capacity waiting can contaminate the
observed training stage; the fit cannot isolate pure GPU time.

## B7 conditions

The local MPS study reuses 15 saved runs from
`docs/results/loss-classifier-local-20261018/inputs/`: healthy, learning rate
0.1x/0.03x, and LoRA rank 2/4, each at seeds 0/1/2. `local-sensitivity.json`
records the source paths and SHA-256 hashes.

The Token Factory study uses the existing 500-example train and 100-example
validation files in `artifacts/b9/data-500/`. Qwen3-0.6B has a three-epoch baseline
and 9/18-epoch variants, each at seeds 0/1/2. Each seed deterministically shuffles
the training rows before upload; the seed's baseline and variants share that file.
Rank and other recipe fields are held fixed. Token Factory rejects rank 2/4, so
rank sensitivity is measured only in the local study.

To reproduce the remote protocol, generate its manifest with:

```bash
uv run python -m scripts.measure_epoch_sensitivity \
  --out artifacts/b467-20261019/tf-epochs \
  --train artifacts/b9/data-500/train.jsonl \
  --valid artifacts/b9/data-500/valid.jsonl
```

`--execute --parallel-seeds 3` runs the paid grid. Do not launch a second runner
against a directory already being processed. Existing job IDs must be reconciled
before resuming after a submission failure. Evaluate each successful final adapter
with `uv run --group eval --group ml python -m scripts.evaluate_epoch_sensitivity`
on the same 100 validation examples
(`--valid artifacts/b9/data-500/valid.jsonl --device mps`; `--watch` waits for jobs). The preliminary local 9/18-epoch training attempt was
interrupted and contributes no results.

Each study reports exact-match means, seed standard deviations and differences
from its own matched healthy seed. Local and provider studies stay separate.
These are descriptive validation curves; no held-out or causal-significance
claim follows, and a low-rank recipe is not assumed to harm quality.

## Agent detection sensitivity

After the training measurements, the frozen agent is evaluated on all 24 saved
cases (15 local, nine provider). Only the observation is passed to the agent:
recipe, checkpoint losses and validation quality against the same-seed reference.
Fault labels, source paths and run names remain with the scorer. Local epoch-end
losses use the recorded training loss at the corresponding validation step.

The measured outcome is the top-ranked recipe class when the existing loop stops
before causal intervention (`causal_checks=False`, `explain=False`, default EIG
policy and threshold 0.6, hypotheses prompt v3). This is detection ranking,
not a causal-confirmation rate. No Tavily calls, repairs or additional training
are performed. Healthy cases measure recipe top-1 false alarms. A missing or
invalid model result counts as a miss, with its error retained.

Each strength has three training seeds and one agent investigation per seed.
The report gives the minimum tested severity whose observed recovery is at least
80%, and separately whether the Wilson 95% lower bound reaches 80%. Three successes
out of three cannot establish the latter. Baselines plus two injected strengths
supply three parameter levels; no untested threshold is interpolated.

The detection stage reserves at most $2.40 (24 cases, $0.10 per case). Completed
training reservations have been replaced by their recorded token-cost estimates
before allocating it; `detection-budget.json` records that reconciliation within
the original $10 cap. Usage is persisted after every completed model call. An
unfinished paid case is not silently retried on resume.
