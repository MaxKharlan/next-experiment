# Next Experiment

A fine-tuning job can finish with status `succeeded` and still produce a model that does not do the task. Next Experiment takes such a job and finds out why. NVIDIA Nemotron ranks the likely causes, the agent runs the cheapest checks that tell them apart, and the leading fix is trained against a control on Nebius Token Factory before anything is retrained.

[Open the demo](https://maxkharlan.github.io/next-experiment/) to replay saved investigations, check training labels in your browser, or start a live run on your own data. No account, API key or payment is needed. The [FT-Fail dataset](https://huggingface.co/datasets/MakHH/FT-Fail) holds the benchmark cases and every method's results.

## Test it without keys or payment

All three ways start from the [demo page](https://maxkharlan.github.io/next-experiment/). The first two never call a model or train anything; the third spends from a shared $20 allowance that the page shows before you start.

### Replay a saved investigation

Press **Replay**. The default case, *Fault 6 on Token Factory: learning rate too low*, plays back in about 10 seconds: four ranked causes, a free probe of the learning rate, the fix trained against a control, the verdict, the approved retrain and the comparison of three models on 200 held-out queries. **Skip to the end** shows the finished report, and **Saved report, JSON** downloads it. The list holds 36 saved runs: 1 agent investigation, 2 measured experiments, 9 local audits and 24 unconfirmed traces.

### Check training labels in your browser

Under *Inspect local training labels*, pick an example, press **Load example**, then **Run audit in this browser**. Edit the observed labels and run it again to see the result change. It checks prompt loss, EOS masking and label alignment.

### Investigate your own data with Nemotron

Press **Run with Nemotron** (or open the [hosted form](https://next.46-17-103-231.sslip.io/) in its own tab). Describe the task and the problem, enter the recipe, and upload four chat JSONL splits: 16 to 64 training rows, 4 to 16 validation rows, and 4 to 8 rows each for check and test. **Download an example with all four files** gives a ready set. Tick the consent box and press **Start with Nemotron**. The run reserves $1, trains a fix and a control with two seeds each on Token Factory, scores the four adapters on CPU, and usually finishes in 10 to 20 minutes when Token Factory has free GPUs. Keep the tab open; a refresh reconnects to the same run, and **Download evidence and report** saves the result.

The hosted runner is deliberately small: Qwen3-0.6B, short answers scored by exact match with at most 32 generated tokens, one diagnostic pair per run and one run at a time. On samples this small the result is exploratory and may be inconclusive. Uploads go to Nebius for training and probe summaries can send excerpts to Nemotron, so do not upload secrets or personal data. [docs/judging.md](docs/judging.md) has the same steps; [docs/hosted-investigations.md](docs/hosted-investigations.md) describes the service and its limits.

To run everything on your own machine instead, see [Run it](#run-it).

## What it does

Given a failed job, Next Experiment works through six steps:

1. Tavily searches for problems others reported with the model and the recipe. Results are cached, and every investigation has a credit limit.
2. Nemotron 3 Super ranks the possible causes from the fault catalog and gives each a probability.
3. Free level-0 probes read the recipe, the loss per epoch and, when the job's adapter is available, its weights. The planner picks the next probe by the expected information it gives per dollar and updates the probabilities from a table of outcome likelihoods.
4. A causal check trains the leading fix and its control as Token Factory jobs, with the seed count the protocol sets. The adapters are downloaded and scored by exact match on a separate check set. The fix counts as confirmed when it beats the control by at least the minimum effect and the 95% interval stays above zero; with three or more seeds per arm, a permutation test must also give p ≤ 0.05.
5. Nemotron 3 Ultra writes the verdict. The code supplies the decision, the evidence and the recipe patch.
6. After you approve the price, one job retrains the fixed recipe on the whole training file, and the base, failed and retrained models answer the same held-out queries.

Every step streams to the screen as it happens. Paid probes, causal checks and the retrain wait for a yes. A supported fix means the fix beats its control; it does not prove that the cause is the only one, and the report says so.

On the saved demo case, Qwen3-0.6B trained with a learning rate of 1e-5 answers 14% of 200 held-out queries. Four Post-training jobs show that 1e-4 beats the unchanged recipe by 0.38 exact match on the check set, and the retrained model answers 42.5%. The run cost about $0.075 in model calls and training.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/media/causal-check-dark.svg">
  <img alt="The causal check of the demo case. The control, the failed recipe trained again, scores 0.12 exact match on both seeds; the fix, a learning rate of 1e-4 instead of 1e-5, scores 0.50 on both. The effect is +0.38 with a 95% interval of 0.35 to 0.41, above the minimum effect of 0.03. With two seeds per arm, the permutation gate is skipped." src="docs/media/causal-check-light.svg" width="100%">
</picture>

## Results

### Benchmark

Four methods diagnosed 13 Token Factory jobs with a known cause: 9 with a learning rate that is too low, in recipes of 3, 6 and 12 epochs at rank 8 or 16, and 4 healthy ones. Thresholds were set on the 5 dev cases; the 8 held-out cases ran once, afterwards.

| Method | Cause ranked first | False alarms | Wrong confirmations | Fix success | $ per case | Jobs per case |
|---|---|---|---|---|---|---|
| Single Ultra call, no experiments | 6/9 | 0/4 | 0 | 0/9 | 0.009 | 0.0 |
| ReAct on Super, same tools | 9/9 | 0/4 | 0 | 6/9 | 0.205 | 4.9 |
| Checklist, every check | 9/9 | 0/4 | 6 | 6/9 | 0.417 | 12.0 |
| Next Experiment | 9/9 | 0/4 | 0 | 6/9 | 0.144 | 3.7 |

Next Experiment finds the same causes as the ReAct agent for 30% less; the saving comes from the healthy cases, where it stops after fewer checks. The checklist also finds every cause, but on 6 of the 9 fault cases it confirms a second, wrong one: a bigger adapter also learns faster at a low learning rate. Every fault case has the same cause, so this benchmark covers one failure family. The held-out cases have since been reused for calibration analysis and are no longer an untouched test set. [docs/benchmark.md](docs/benchmark.md) has both splits case by case and the error analysis; the cases and the 52 method rows are published as [FT-Fail](https://huggingface.co/datasets/MakHH/FT-Fail) with source licenses and injection manifests.

### Weight probe

Reading the LoRA adapter's weights separates a too-low learning rate from healthy jobs on Token Factory: with a threshold set on the dev cases, 9 of 9 held-out jobs and 84 of 86 causal-check adapters land on the right side ([docs/b25-lora-weights.md](docs/b25-lora-weights.md)).

### Answers cut before upload

A second fault family on BANKING77 and Qwen3-0.6B, scored on 770 check examples. The target drops the answers the probe flags; the control drops as many random unflagged ones and trains about 9% more tokens.

| Environment | Healthy | Damaged | Target | Control | Target minus control |
|---|---:|---:|---:|---:|---:|
| Kaggle T4, 3 seeds per arm | 0.883 | 0.658 | 0.804 | 0.621 | +0.183 |
| Token Factory, 2 seeds per arm | 0.926 | 0.681 | 0.859 | 0.662 | +0.197 |

Recovery is partial, and these saved experiments have no end-to-end agent trace ([docs/answer-cut.md](docs/answer-cut.md)).

### Reference models

On Nebius AI Cloud, 5,012 saved answers were checked for Qwen3-0.6B and 1.7B adapters chosen on validation. They scored 104/154 and 115/154 on BANKING77, 100/102 and 102/102 on function calling, and 102/102 each on JSON extraction. The 1.7B base model already scores 100% on both structured tasks ([docs/qwen-reference-quality-20261020.md](docs/qwen-reference-quality-20261020.md)). One pinned base checkpoint gave identical answers on all 64 development queries on a MacBook (MPS), an AI Cloud L40S and a Kaggle T4 ([docs/kaggle-runtime-comparison-20261020.md](docs/kaggle-runtime-comparison-20261020.md)).

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/media/architecture-dark.svg">
  <img alt="Architecture. The screen in the browser talks to the API. The API, the investigation worker with its agent loop, model router, planner and probes, the causal check and storage run in one Python process. The worker calls Tavily and Nemotron 3 Super and Ultra; the causal check trains LoRA jobs on the Post-training API of Nebius Token Factory. Numbers 1 to 6 mark the steps of one investigation." src="docs/media/architecture-light.svg" width="100%">
</picture>

The animation shows one investigation in one API worker; its numbers are the six steps above.

| Part | Where it runs | Code | What it does |
|---|---|---|---|
| Screen | the browser; the API serves it, and GitHub Pages serves a static copy that replays saved runs | `apps/api/static/` | starts a replay or a live run, streams each step, asks before paid steps |
| API | a uvicorn process | `apps/api/` | creates investigations, streams their events over server-sent events, records paid-step decisions, serves reports |
| Investigation worker | a thread in the API process, one per investigation | `core/agent/`, `core/planner/`, `core/probes/` | runs the agent loop, sends each step to its model, picks the next probe or check |
| Causal check | the same thread; jobs on Token Factory, or local training with the `local` backend | `core/interventions/`, `core/eval/`, `jobs/eval_local/` | trains fix and control pairs, downloads the adapters, scores them, applies the decision rule |
| Hosted live runs | a container on a VPS; the demo page embeds its form | `apps/judge/`, `deploy/judge/` | bounded live investigations on uploaded data from the shared $20 allowance |
| Remote scoring | Nebius AI Cloud Serverless Jobs, Kaggle | `core/connectors/compute.py`, `jobs/cloud/`, `jobs/kaggle/` | scores adapters on a GPU when the local machine is too slow |
| State and artifacts | SQLite; local files or Nebius Object Storage | `apps/api/store.py`, `core/db/`, `core/connectors/storage.py` | events, execution leases, spending reservations, check results, the Tavily cache |
| Models and training | Nebius Token Factory | `core/connectors/token_factory.py` | Nemotron 3 Super and Ultra, the Post-training API |
| Known issues | Tavily | `core/connectors/tavily.py` | searches for problems others reported with the model and the recipe |

The default state backend allows one API process; with SQLite state several API workers on one host share a database, and execution leases keep two workers from paying for the same step. Finished investigations and pending approvals survive a restart, and a paid step whose outcome is unknown keeps its budget reservation until an operator checks it. Every model call is recorded with its usage and, when the provider returns one, its request ID. [docs/architecture.md](docs/architecture.md) follows an investigation through the code.

## Run it

You need Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Copy `.env.example` to `.env`. Replays need no keys. Live runs need `TOKEN_FACTORY_API_KEY`, and the known-issues step needs `TAVILY_API_KEY`.

Start the screen, open http://127.0.0.1:8000 and press Replay:

```bash
uv run uvicorn apps.api.main:app
```

Live runs train on Token Factory and score adapters locally, so they need the ML groups. With `PUBLIC_DEMO=true` the server allows only replays.

```bash
uv run --group ml --group eval uvicorn apps.api.main:app
```

Several API workers on one host need SQLite state; read [docs/sqlite-investigations.md](docs/sqlite-investigations.md) before changing an existing deployment:

```bash
INVESTIGATION_BACKEND=sqlite DATABASE_URL=sqlite:///./artifacts/next_experiment.db uv run --group ml --group eval uvicorn apps.api.main:app --workers 2
```

Investigate the demo case from the command line, with causal checks on Token Factory (about $0.06 and 10 to 20 minutes per check when Token Factory has free capacity). `--job <failed job ID>` also runs the weight probe on that job's adapter:

```bash
uv run --group ml --group eval python -m scripts.investigate_tf
```

Retrain after a confirmed investigation; it shows the price and asks before it starts:

```bash
uv run --group ml --group eval python -m scripts.retrain_tf
```

Rerun the dev benchmark from the cached pairs, without training:

```bash
uv run --group ml --group eval python -m bench.run_bench --split dev --no-train --check-set artifacts/b10/check.jsonl --noise-sd 0.019 --min-effect 0.06
```

Long Token Factory runs fill the account's limit of 5000 files with checkpoints. Keep this running next to them; it keeps each finished job's last checkpoint and deletes the others:

```bash
uv run python -m scripts.tf_prune_checkpoints --delete --watch 10
```

Build the static demo that `.github/workflows/pages.yml` publishes on every push to main, then serve it on http://127.0.0.1:8000:

```bash
uv run python -m scripts.build_pages --out _site
```

```bash
python -m http.server -d _site
```

Run the tests (CI runs ruff and pytest on every pull request and push to main):

```bash
uv run --group eval pytest -q
```

## Repository layout

| Path | What is there |
|---|---|
| `apps/api/` | HTTP API, event stream, investigation store, case validation and the screen (`static/`) |
| `apps/judge/`, `deploy/judge/` | the hosted live-run service, its form and its container |
| `core/agent/` | investigation loop, model router, prompts, schemas, known issues, label checks |
| `core/planner/` | expected information gain, stopping rule, outcome likelihoods, job cost model |
| `core/probes/` | level-0 probes (recipe, lengths, answers, overlap, distribution, adapter weights), level-1 logit probes and the signature table |
| `core/interventions/` | fix and control pairs for recipe, data and trainer faults, causal-check statistics, the Token Factory runner, retrain |
| `core/eval/`, `core/training/` | checkpoint scoring, metrics and error breakdowns; gradient and trainable-parameter audits |
| `core/connectors/` | Token Factory, Tavily, storage and Serverless Jobs |
| `core/db/`, `core/telemetry/`, `core/report/` | spending and credit ledgers, per-call usage, investigation and billing reports |
| `bench/` | tasks, cases, fault injections, comparison methods, fixes, references, scorer and runner |
| `datasets/` | the JSON-extraction and function-calling tasks with their splits and manifests |
| `jobs/` | local adapter scoring, Kaggle notebooks, AI Cloud jobs and the opt-in Docker sandbox for generated checks |
| `scripts/` | command-line tools: investigations, retrain, sweeps, cases, calibration, the static demo |
| `demo/` | the saved investigations and audits the screen replays |
| `docs/` | write-ups and dated result records; `docs/results/` holds the raw evidence |
| `docs/animation/` | the README animations, drawn with [ManimGX](https://github.com/academa-labs/manimgx) and written as SVG by its `svg.py` (MIT) |
| `tests/` | unit, end-to-end API and browser tests |

## Documents

| File | Contents |
|---|---|
| [docs/judging.md](docs/judging.md) | how to test the demo, the live runs and what they spend |
| [docs/architecture.md](docs/architecture.md) | how an investigation moves through the code: model steps, modules, contracts, API, storage, configuration |
| [docs/benchmark.md](docs/benchmark.md) | dev and held-out results case by case, error analysis, limits |
| [docs/project-status.md](docs/project-status.md) | what is covered, what is open, and the evidence for each |
| [docs/hosted-investigations.md](docs/hosted-investigations.md) | the hosted service: limits, budget guard, deployment checks |
| [docs/answer-cut.md](docs/answer-cut.md) | the second fault family: protocol and Kaggle and Token Factory results |
| [docs/b25-lora-weights.md](docs/b25-lora-weights.md) | the weight probe: measurements, threshold, why rank is not used |
| [docs/b10-noise-floor.md](docs/b10-noise-floor.md) | noise between seeds and the decision rule of the causal check |
| [docs/b6-post-training.md](docs/b6-post-training.md) | what a Token Factory LoRA job costs, how long it takes and what it returns |
| [docs/sqlite-investigations.md](docs/sqlite-investigations.md), [docs/paid-operation-recovery.md](docs/paid-operation-recovery.md) | API workers, leases, budget reservations and unknown paid outcomes |
| [docs/feedback.md](docs/feedback.md) | what we ran into on Token Factory and Nebius AI Cloud, with dates |

## Notes on Token Factory

- Training is deterministic: the same files and hyperparameters give a byte-identical adapter, and the `seed` parameter has no effect. Repeats differ only when the training file is shuffled differently.
- `result_files` hold a PEFT adapter whose weight names lack the `base_model.model.` prefix, so `PeftModel.from_pretrained` loads none of them and keeps random weights with only a warning. `jobs/eval_local/exact_match.py` loads them correctly.
- A job waiting for a GPU shows 0 steps and no queue status, so it looks stalled.
- Every checkpoint counts against the 5000-file limit, one checkpoint per epoch.

[docs/feedback.md](docs/feedback.md) has the details and the dates.

## Limits

- Every fault case in the agent benchmark has the same cause, a learning rate that is too low. In our sweeps the other planted recipe faults, a small LoRA rank and too many epochs, did not hurt this task.
- The agent benchmark uses BANKING77 and Qwen3-0.6B with two seeds per arm. The answer-cut experiments add a second fault family on the same task and model.
- Training costs in the tables are estimates from token counts and Token Factory's price per million tokens, not invoices.
- The hosted runner supports one model, short exact-match answers and one diagnostic pair per run.

## Task labels

Code comments, docs and pull requests name tasks of the project plan by a letter and a number. The plan has two tracks: A covers data and model behaviour, B covers training and the agent's decisions. The tables list the tasks that appear in this repository.

| Task | What it is | Where |
|---|---|---|
| A1 | BANKING77 splits for generation with a manifest of file hashes | `bench/tasks/banking77.py`, `docs/banking77.md` |
| A2 | JSON field extraction task: schema, examples, valid-JSON and field metrics | `bench/tasks/structured.py`, `datasets/structured-mixed-v1/json_extraction/` |
| A3 | function-calling task: function specs, queries, reference calls | `bench/tasks/structured.py`, `datasets/structured-mixed-v1/function_calling/` |
| A4 | dataset validation before upload: message format, roles, empty answers | `apps/api/case_input.py` |
| A5 | data faults 1 to 5 as deterministic changes with a strength and a seed | `bench/faults/data.py` |
| A7 | targeted fixes and equal-cost controls for data faults | `core/interventions/data_pairs.py` |
| A8 | example lengths in tokens against `context_length` | `core/probes/level0/lengths.py` |
| A9 | chat template check against the base model's template | `docs/remaining-scope-20261020.md` (partly measured) |
| A10 | overlap between train and validation | `core/probes/level0/overlap.py` |
| A11 | contradictory labels: embedding neighbours with different answers, audited by Nemotron Nano | `core/probes/level0/semantic.py` |
| A12 | distribution shift: label divergence and an embedding two-sample check | `core/probes/level0/distribution.py` |
| A13 | evaluation module: checkpoint inference, metrics, error breakdown | `core/eval/` |
| A14 | signature-table rows and calibration for track A's probes | `core/probes/signatures.yaml` |
| A15 | evaluation as Nebius AI Cloud Serverless Jobs | `core/connectors/compute.py`, `jobs/cloud/` |
| A18 | sensitivity curves for data faults at several strengths | `docs/data-fault-quality-20261020.md` (partial) |
| A19 | Nemotron Nano as a labeller on 200 examples, against Super | `docs/semantic-audit-20261020.md` |
| A20 | one checkpoint scored on a MacBook, AI Cloud and Kaggle | `docs/kaggle-runtime-comparison-20261020.md` |
| A24 | benchmark runner with equal budgets and intervals | `bench/run_bench.py` |
| A25 | reproduction of public cases with data errors | `docs/public-finetuning-cases.md` |
| A27 | downloadable investigation report | `core/report/investigation.py` |
| A29 | logit probes on checkpoints: answer and prompt loss, EOS probability | `core/probes/level1/`, `scripts/probe_logits.py` |
| A30, A31 | local embeddings and a cleanlab baseline for noisy labels | `docs/semantic-audit-20261020.md` |
| A32 | the FT-Fail dataset on Hugging Face | `docs/ft-fail-hub/` |

| Task | What it is | Where |
|---|---|---|
| B1 | Token Factory connector: structured Nemotron output, files, jobs, events, checkpoints | `core/connectors/token_factory.py` |
| B2 | router of steps to Nemotron models and the call log | `core/agent/router.py`, `core/telemetry/` |
| B3 | Tavily connector with a cache and credit limits | `core/connectors/tavily.py` |
| B4 | running checks the agent writes, in an isolated sandbox | `core/agent/sandbox_check.py`, `jobs/sandbox/` |
| B5 | reference runs: the best recipe for a task and model | `bench/references/`, `scripts/reference_tf.py` |
| B6 | cost and duration model of Post-training jobs | `core/planner/job_cost.py`, `docs/b6-post-training.md` |
| B7 | planted recipe faults 6 to 8 | `bench/faults/recipe.py` |
| B8 | fixes and equal-cost controls for faults 6 to 8 | `core/interventions/pairs.py` |
| B9 | simulation lab: fault sweeps locally and on Token Factory | `jobs/kaggle/`, `scripts/tf_fault_sweep.py` |
| B10 | noise floor between seeds | `scripts/noise_floor.py`, `docs/b10-noise-floor.md` |
| B11 | statistics and decision rule of the causal check | `core/interventions/stats.py` |
| B12 | calibration of Super's probabilities: reliability, ECE, Brier, temperature | `bench/calibration.py`, `docs/confidence-calibration.md` |
| B13 | versioned prompts | `core/agent/prompts/` |
| B14 | several Super samples per hypothesis step, aggregated | `core/agent/ensemble.py`, `docs/hypothesis-sampling.md` |
| B15 | known issues from Tavily | `core/agent/known_issues.py` |
| B16 | planner | `core/planner/` |
| B17 | investigation loop with budget and paid-step approval | `core/agent/loop.py` |
| B18 | verdict and patch, retrain after approval, comparison of three models | `core/interventions/retrain.py`, `scripts/retrain_tf.py` |
| B19 | comparison methods for the benchmark | `bench/baselines/` |
| B20 | ablations: without Tavily, planner or causal check | `bench/ablations.py`, `docs/ablations.md` |
| B21 | public reports of fine-tunes that finished but did not work | `bench/public_cases/`, `docs/public-finetuning-cases.md` |
| B22 | HTTP API | `apps/api/` |
| B23 | investigation screen | `apps/api/static/index.html` |
| B24 | trainer faults 9 to 12 and their code checks | `core/interventions/code_faults.py`, `core/training/` |
| B25 | weight probe on the LoRA adapter | `core/probes/level0/weights.py`, `docs/b25-lora-weights.md` |
| B26 | classifier of loss-curve signatures from the B9 sweeps | `bench/loss_classifier.py`, `docs/loss-curve-classifier.md` |

## License

Apache 2.0, see `LICENSE`.
