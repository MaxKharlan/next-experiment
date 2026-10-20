# Next Experiment

An agent that investigates LLM fine-tune failures and tests proposed fixes against controls on Nebius Token Factory.

Demo: [maxkharlan.github.io/next-experiment](https://maxkharlan.github.io/next-experiment/) replays a saved investigation in your browser. The video below is that page, slowed down so that each step can be read.

https://github.com/user-attachments/assets/d46683f9-7c1d-46d0-bf82-f413ff1851ee

## What it does

A fine-tuning job can finish with status `succeeded` and still give a model that does not do the task. Next Experiment takes such a job, finds the likely causes, and runs the cheapest experiments that can tell them apart. A controlled check supports a fix when it beats the specified control beyond the measured noise between seeds. That comparison alone does not establish a unique root cause. Controls must report both retained examples and training tokens; equal file size does not imply equal compute cost. After you approve the cost, it retrains the model with the fix and compares the base, failed and retrained models on the same held-out queries.

On the saved demo case, a Qwen3-0.6B job trained with a learning rate of 1e-5 answers 14% of 200 held-out queries. Four Post-training jobs support increasing the learning rate against the unchanged-recipe control, and the retrained model answers 42.5%. The whole run, model calls and training, costs about $0.075.

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/media/architecture-dark.svg">
  <img alt="Architecture. The screen in the browser talks to the API. The API, the investigation worker with its agent loop, model router, planner and probes, the causal check and storage run in one Python process. The worker calls Tavily and Nemotron 3 Super and Ultra; the causal check trains LoRA jobs on the Post-training API of Nebius Token Factory. Numbers 1 to 6 mark the steps of one investigation." src="docs/media/architecture-light.svg" width="100%">
</picture>

Next Experiment runs on Nebius Token Factory: Nemotron 3 Super and Ultra do the reasoning, and the Post-training API trains the fix and control jobs of each causal check and the final retrain.

The API starts a background thread for each investigation and scores downloaded adapters locally. The default state backend allows one API process; SQLite state supports multiple API workers on one host, sharing a database, working directory and artifacts. Artifact storage can use a local folder or Nebius Object Storage with `STORAGE_BACKEND=s3`; execution leases and spending ledgers stay in local SQLite files.

The animation shows one investigation in a single API worker; its numbers are the steps in the next section.

| Part | Where it runs | Code | What it does |
|---|---|---|---|
| Screen | the browser; the API serves it as one static page, and GitHub Pages serves a copy that replays the saved run without a server | `apps/api/static/index.html` | starts a replay or a live run, shows each step as it streams in, asks before paid probes, causal checks and the final retrain |
| API | a uvicorn process on your machine | `apps/api/` | creates investigations, streams their events over server-sent events, records paid-step decisions and serves downloadable reports |
| Investigation worker | a thread in the same process, one for each investigation | `core/agent/`, `core/planner/`, `core/probes/` | runs the agent loop, sends each step to its model, picks the next probe or check |
| Causal check | the same thread; jobs on Token Factory, or local training with the `local` backend; scoring on the app's machine with CUDA, Apple MPS or the CPU | `core/interventions/`, `jobs/eval_local/` | trains fix and control pairs, downloads the adapters, scores them by exact match, applies the decision rule |
| State and artifacts | SQLite for shared state; local files or Nebius Object Storage for artifacts | `apps/api/store.py`, `core/db/`, `core/connectors/storage.py` | metadata, events, execution ownership, spending reservations, check results and the Tavily cache |
| Models and training | Nebius Token Factory | `core/connectors/token_factory.py` | Nemotron 3 Super and Ultra, the Post-training API |
| Known issues | Tavily | `core/connectors/tavily.py` | searches for problems others reported with the model and the recipe |

Finished investigations and pending paid-step approvals survive a restart. SQLite workers use execution leases and request deduplication; expired work becomes failed for operator review. Unknown paid outcomes retain budget reservations. A separate worker queue, multi-host deployment, a Next.js screen and Hugging Face Spaces hosting remain unimplemented. See [API worker coordination](docs/sqlite-investigations.md) and [paid-operation recovery](docs/paid-operation-recovery.md).

## How an investigation runs

1. Tavily searches for problems others reported with the model and recipe. Results are cached, and every investigation has a credit limit.
2. Nemotron 3 Super ranks the possible causes from the fault catalog and gives each a probability.
3. Free level-0 probes read the recipe, the loss per epoch and, when the job's adapter is available, the adapter's weights. The planner picks the next probe by the expected information it gives per dollar and updates the probabilities from a table of outcome likelihoods.
4. A causal check tests the leading proposed fix against its configured control, using the protocol's seed count. The adapters are downloaded and scored locally by exact match on a separate check set. The stored `confirmed` decision means the fix beats the control by at least the minimum effect and the 95% interval is above zero; with three or more seeds per arm, a permutation test must also give p ≤ 0.05.
5. Nemotron 3 Ultra writes the verdict explanation. The code supplies the intervention decision, evidence and recipe patch.
6. With your approval, one job retrains the fixed recipe on the whole training file, and the three models answer the same held-out queries.

Each step is streamed to the screen as it happens. Paid probes and causal checks pause for approval before submission. Reports retain per-attempt model usage and available provider request/response IDs linked to budget reservations; see [persisted LLM usage](docs/persisted-llm-usage.md).

Step 4 on the demo case: the fix and the control train two jobs each, and the effect stands far outside the noise between seeds.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/media/causal-check-dark.svg">
  <img alt="The intervention check of the demo case. The control, the failed recipe trained again, scores 0.12 exact match on both seeds; the fix, a learning rate of 1e-4 instead of 1e-5, scores 0.50 on both. The effect is +0.38 with a 95% interval of 0.35 to 0.41, above the minimum effect of 0.03. With two seeds per arm, the permutation gate is skipped." src="docs/media/causal-check-light.svg" width="100%">
</picture>

## Results

The benchmark compares four methods on 13 Token Factory jobs with a known cause: 9 with a learning rate that is too low, in recipes of 3, 6 and 12 epochs at rank 8 or 16, and 4 healthy ones. The thresholds were set on the 5 dev cases; the 8 held-out cases ran once, after that.

| method | top-1 | false alarms | wrong confirmations | fix success | $ per case | jobs per case |
|---|---|---|---|---|---|---|
| Single Ultra (one call, no experiments) | 6/9 | 0/4 | 0 | 0/9 | 0.009 | 0.0 |
| ReAct on Super, same tools | 9/9 | 0/4 | 0 | 6/9 | 0.205 | 4.9 |
| Checklist, every check | 9/9 | 0/4 | 6 | 6/9 | 0.417 | 12.0 |
| Next Experiment | 9/9 | 0/4 | 0 | 6/9 | 0.144 | 3.7 |

On these historical low-learning-rate cases, Next Experiment ranks the same planted fault as the ReAct agent at 30% lower cost; the saving comes from the healthy cases, where it stops after fewer checks. The checklist also finds every cause, but it proves a second, wrong cause on 6 of the 9 fault cases. `docs/benchmark.md` has both splits, the results case by case, the error analysis and the limits.

The weight probe (B25) separates a too-low learning rate from healthy jobs on Token Factory: with a threshold set on the dev cases, 9 of 9 held-out jobs and 84 of 86 causal-check adapters land on the right side (`docs/b25-lora-weights.md`).

The historical held-out cases have since been reused for diagnosis and calibration
analysis. They are no longer an untouched evaluation set. Random-order ablations
also confirm a small-rank intervention on some low-learning-rate cases: a useful
change can be attributed to the wrong cause. See [the ablation results](docs/results/ablations-dev-20261018/summary.json).

Answer cuts before upload now have measured results on BANKING77 / Qwen3-0.6B:

| Environment | Healthy | Damaged | Target | Control | Target minus control |
|---|---:|---:|---:|---:|---:|
| Kaggle T4, 3 seeds per arm | 0.883 | 0.658 | 0.804 | 0.621 | +0.183 |
| Token Factory, 2 seeds per target/control arm | 0.926 | 0.681 | 0.859 | 0.662 | +0.197 |

These are scores on 770 check examples. The target drops flagged answers and the
control drops the same number of random unflagged examples. The control trains
about 9% more tokens. Recovery is partial; these saved experiments contain no
end-to-end agent trace. Token Factory healthy/damaged references have one job
each. The two-seed permutation gate cannot reach 0.05 and is skipped by the saved
decision rule. See [the experiment protocol and results](docs/answer-cut.md).

## Run it

You need Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Copy `.env.example` to `.env`. Replaying the saved investigation needs no keys. Live runs need `TOKEN_FACTORY_API_KEY`, and the known-issues step needs `TAVILY_API_KEY`.

Replay the demo on the screen, then open http://127.0.0.1:8000 and press Replay:

```bash
uv run uvicorn apps.api.main:app
```

With `PUBLIC_DEMO=true` the server allows only replays. Live runs from the screen train on Token Factory and score adapters locally, so they need the ML groups:

```bash
uv run --group ml --group eval uvicorn apps.api.main:app
```

For multiple API workers on one host, use SQLite state from the repository root:

```bash
INVESTIGATION_BACKEND=sqlite \
DATABASE_URL=sqlite:///./artifacts/next_experiment.db \
uv run --group ml --group eval uvicorn apps.api.main:app --workers 2
```

Live starts on this backend require `Idempotency-Key`; paid-step and retrain
answers require the waiting event's `proposal_seq`. The screen sends both.
Read [deployment and migration instructions](docs/sqlite-investigations.md)
before changing an existing deployment.

The public demo is the same screen built as a static site; `.github/workflows/pages.yml` publishes it to GitHub Pages on pushes to main. Build it, serve it, then open http://127.0.0.1:8000:

```bash
uv run python -m scripts.build_pages --out _site
```

```bash
python -m http.server -d _site
```

The README's video is the same site, recorded at a slower pace so that each step can be read. Recording it needs Google Chrome and ffmpeg:

```bash
uv run --with playwright python -m scripts.record_demo_video --out demo.mp4
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

Long runs on Token Factory fill the account's limit of 5000 files with checkpoints. Keep this next to them; it keeps each finished job's last checkpoint and deletes the others:

```bash
uv run python -m scripts.tf_prune_checkpoints --delete --watch 10
```

Run the tests:

```bash
uv run --group eval pytest -q
```

## Repository layout

| Path | What is there |
|---|---|
| `apps/api/` | HTTP API, event stream, investigation store and the screen (`static/index.html`) |
| `core/agent/` | investigation loop, model router, prompts, schemas, known issues |
| `core/planner/` | expected information gain, stopping rule, outcome likelihoods, job cost model |
| `core/probes/` | level-0 probes and the signature table |
| `core/interventions/` | fix and control pairs, statistics of the causal check, Token Factory runner, retrain |
| `core/connectors/` | Token Factory, Tavily and storage |
| `core/telemetry/` | per-attempt usage and provider request references |
| `core/db/` | spending and Tavily credit ledgers |
| `core/report/`, `core/eval/` | investigation and billing reports, dataset audits and local evaluation |
| `bench/` | cases, fault recipes, comparison methods, fixes, references, scorer and runner |
| `jobs/eval_local/` | exact-match scoring of adapters on a Mac |
| `jobs/kaggle/` | LoRA training for the local simulation sweeps |
| `scripts/` | command-line tools: investigations, retrain, sweeps, cases, calibration |
| `demo/` | the saved investigation the screen replays |
| `docs/` | write-ups listed below |
| `docs/animation/` | the scenes of the README's animations, the architecture and the causal check, drawn with [ManimGX](https://github.com/academa-labs/manimgx) and written as SVG by its `svg.py` (MIT) |

## Documents

| File | Contents |
|---|---|
| `docs/project-status.md` | current coverage, owners and remaining acceptance criteria |
| `docs/sqlite-investigations.md` | same-host API workers, leases, request identity and migration |
| `docs/paid-operation-recovery.md` | budget reservations and unknown paid outcomes |
| `docs/persisted-llm-usage.md` | per-attempt usage and provider request IDs |
| `docs/provider-cost-reconciliation-20261019.md` | provider Usage reconciliation and outstanding invoice evidence |
| `docs/fault-1-semantics.md` | pre-upload answer damage versus custom-trainer truncation and length preflight |
| `docs/demo-walkthrough.md` | replay walkthrough and boundaries for the O2 demo |
| `docs/architecture.md` | how an investigation moves through the code: model steps, modules, contracts, API, storage, configuration |
| `docs/benchmark.md` | dev and held-out results, case by case, error analysis, limits |
| `docs/b25-lora-weights.md` | the weight probe: measurements, threshold, why rank is not used |
| `docs/b10-noise-floor.md` | noise between seeds and the decision rule of the causal check |
| `docs/b6-post-training.md` | what a Token Factory LoRA job costs, how long it takes and what it returns |
| `docs/feedback.md` | what we ran into on Token Factory and Nebius AI Cloud, dated |

## Notes on Token Factory

- Training is deterministic: the same files and hyperparameters give a byte-identical adapter, and the `seed` parameter has no effect. Repeats differ only when the training file is shuffled differently.
- `result_files` hold a PEFT adapter, but the weight names lack the `base_model.model.` prefix, so `PeftModel.from_pretrained` loads none of them and leaves random weights with only a warning. `jobs/eval_local/exact_match.py` loads them correctly.
- A job waiting for a GPU shows 0 steps and no queue status, so it looks stalled.
- Every checkpoint counts against the 5000-file limit, one checkpoint per epoch.

`docs/feedback.md` has the details and the dates.

## Limits

- Every fault case in the benchmark has the same cause, a learning rate that is too low. In our sweeps the other planted faults, a small LoRA rank and too many epochs, did not hurt this task.
- Historical agent benchmark: BANKING77 and Qwen3-0.6B with two seeds per arm. Answer-cut experiments add a second fault family on the same task/model, with three Kaggle seeds and two Token Factory seeds per intervention arm.
- Training costs in the tables are estimates from token counts and Token Factory's price per million tokens.

## Task labels

Code comments, docs and pull requests name tasks of the project plan by a letter and a number:

| Label | Meaning |
|---|---|
| B<i> | task number i of track B of the plan. This repository holds track B's work. |
| A<i> | task number i of track A. A few comments name one where the code here stands in for that task or leaves room for it. |

The B tasks that appear here:

| Task | What it is | Where |
|---|---|---|
| B1 | Token Factory connector: structured Nemotron output, files, jobs, events, checkpoints | `core/connectors/token_factory.py` |
| B2 | router of steps to Nemotron models and the call log | `core/agent/router.py`, `core/telemetry/` |
| B3 | Tavily connector with a cache and credit limits | `core/connectors/tavily.py` |
| B5 | reference runs: the best recipe for a task and model | `bench/references/`, `scripts/reference_tf.py` |
| B6 | cost and duration model of Post-training jobs | `core/planner/job_cost.py`, `docs/b6-post-training.md` |
| B7 | planted faults 6 to 8 in a recipe | `bench/faults/recipe.py` |
| B8 | fixes and specified control recipes | `core/interventions/pairs.py` |
| B9 | simulation lab: fault sweeps locally and on Token Factory | `jobs/kaggle/`, `scripts/tf_fault_sweep.py` |
| B10 | noise floor between seeds | `scripts/noise_floor.py`, `scripts/rescore_pairs.py`, `docs/b10-noise-floor.md` |
| B11 | statistics and decision rule of the causal check | `core/interventions/stats.py` |
| B13 | versioned prompts | `core/agent/prompts/` |
| B15 | known issues from Tavily | `core/agent/known_issues.py` |
| B16 | planner | `core/planner/` |
| B17 | investigation loop | `core/agent/loop.py` |
| B18 | verdict and patch, retrain after approval, comparison of three models | `core/interventions/retrain.py`, `scripts/retrain_tf.py` |
| B19 | comparison methods for the benchmark | `bench/baselines/` |
| B22 | HTTP API | `apps/api/` |
| B23 | investigation screen | `apps/api/static/index.html` |
| B25 | weight probe on the LoRA adapter | `core/probes/level0/weights.py`, `docs/b25-lora-weights.md` |

## License

Apache 2.0, see `LICENSE`.
