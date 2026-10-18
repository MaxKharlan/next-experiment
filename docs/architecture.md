# Architecture

Next Experiment is one Python process on the developer's machine. It serves the investigation screen and the HTTP API, runs each investigation in a background thread, and scores downloaded adapters on its own GPU or CPU. Nemotron 3 Super and Ultra on Nebius Token Factory do the reasoning, and Token Factory's Post-training API trains the jobs of every causal check and the final retrain. Tavily supplies known problems. Artifacts stay in a local folder or in an Object Storage bucket on Nebius AI Cloud. A static copy of the screen on GitHub Pages replays a saved investigation without the process.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/architecture-dark.svg">
  <img alt="Architecture. The screen in the browser talks to the API. The API, the investigation worker with its agent loop, model router, planner and probes, the causal check and storage run in one Python process. The worker calls Tavily and Nemotron 3 Super and Ultra; the causal check trains LoRA jobs on the Post-training API of Nebius Token Factory. Numbers 1 to 6 mark the steps of one investigation." src="media/architecture-light.svg" width="100%">
</picture>

The README's [Architecture](../README.md#architecture) section has the table of where each part runs. This page goes one level down: how an investigation moves through the code, which model does which step, the modules and their contracts, the API, storage, configuration and checks.

## The path of one investigation

An investigation passes through the statuses `queued`, `running`, `awaiting_approval` and `finished`, or ends in `failed` on an error.

1. The screen sends `POST /api/investigations` with the backend `replay`, `local` or `tf`. The API saves the record as `queued` and starts a thread for it. With `PUBLIC_DEMO=true` it accepts only `replay`.
2. The thread emits `started`, and the status becomes `running`. Every event goes into memory and into `investigations/<id>/events.jsonl`, and `GET /api/investigations/{id}/events` streams it to the screen.
3. `core/agent/known_issues.py` builds two Tavily queries from the model name, the recipe and the symptom, never from dataset examples, and repeated queries come from the cache in `tavily-cache/`. Super reads the results with the fault catalog and proposes hypotheses with priors and sources.
4. The planner picks the level-0 probe with the most expected information per dollar. The probe returns a discrete outcome, the code updates the probabilities from `core/probes/signatures.yaml`, and Super writes the probe's card.
5. When the leading hypothesis passes the threshold, `core/interventions/pairs.py` builds the fix and a control that costs at least as much. On the `tf` backend, `TokenFactoryPairRunner` trains each arm as two Post-training jobs, one per seed, downloads the adapters and scores them by exact match on the check set. On the `local` backend the arms train on the machine itself (`jobs/kaggle/run_pair.py`).
6. `causal_check` confirms the cause when the effect reaches the minimum effect and the 95% interval is above zero; with three or more seeds per arm, a permutation test must also give p ≤ 0.05. Ultra writes the verdict, and the code sets the status and the patch.
7. On the `tf` backend a confirmed cause ends with a proposal to retrain and its price, and the status becomes `awaiting_approval`. The answer comes through `POST /api/investigations/{id}/retrain`.
8. After approval, one job trains the fixed recipe on the whole training file, and the base, failed and fixed models answer the same held-out queries. The status becomes `finished`. After a restart, finished investigations load from storage.

Steps 5 and 6 on the demo case:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/causal-check-dark.svg">
  <img alt="The causal check of the demo case. The control, the failed recipe trained again, scores 0.12 exact match on both seeds; the fix, a learning rate of 1e-4 instead of 1e-5, scores 0.50 on both. The effect is +0.38 with a 95% interval of 0.35 to 0.41, well above the minimum effect of 0.03, so the cause is confirmed." src="media/causal-check-light.svg" width="100%">
</picture>

## Model steps

The router in `core/agent/router.py` sends each step to one Nemotron 3 model. Once Super has twice failed to square its hypotheses with the probe results, the `hypotheses` and `interpret_probe` steps move to Ultra. Every call goes into the call log with its tokens, cost, latency and prompt version.

| Step | Model | Who calls it |
|---|---|---|
| `hypotheses` | Super | the loop, step 3 |
| `interpret_probe` | Super | the loop, step 4 |
| `verdict` | Ultra | the loop, step 6 |
| `react` | Super | the ReAct comparison method in `bench/baselines/` |
| `single_shot` | Ultra | the single-Ultra comparison method |
| `audit_example`, `label_errors`, `parse_events` | Nano | nothing yet; reserved for calls on each example |
| `known_issues`, `probe_code` | Super | nothing yet; the known-issues step writes its queries in code |
| `patch` | Ultra | nothing yet; the code writes the patch |

The prompts are versioned files in `core/agent/prompts/`, such as `hypotheses.v2.md`.

## Modules

| Module | What it does | Entry points |
|---|---|---|
| `core/config.py` | settings from the environment | `get_settings()` |
| `core/case.py` | the input of an investigation: task, symptom, job, four data splits, metric, budget | `InvestigationCase` |
| `core/agent/loop.py` | the loop: known issues, hypotheses, probes, causal check, verdict, retrain proposal | `investigate(ctx, case, router, run_pair, probes=…, on_step=…)` |
| `core/agent/router.py` | the model for each step and the move to Ultra | `Router.structured(step, …)`, `Router.chat(step, …)` |
| `core/agent/steps.py` | the model steps and versioned prompts | `propose_hypotheses`, `interpret_probe`, `write_verdict`, `load_prompt` |
| `core/agent/known_issues.py` | Tavily queries and the sources hypotheses may cite | `find_known_issues(…)` |
| `core/agent/schemas.py` | what the models must return | `Hypothesis`, `HypothesisList`, `ProbeInterpretation`, `Verdict` |
| `core/planner/outcome_model.py` | outcome probabilities from the signature table | `likelihood(probe, fault_class, outcome)` |
| `core/planner/eig.py` | entropy, the Bayes update, expected information per dollar | `update(…)`, `expected_gain(…)`, `best_probe(…)` |
| `core/planner/stopping.py` | the next action: a probe, a causal check or a stop | `next_action(…)` |
| `core/planner/job_cost.py` | price and duration of Post-training jobs | `estimate_job(…)`, `price_per_m(model)` |
| `core/probes/base.py` | the probe contract | `Probe`, `ProbeResult` |
| `core/probes/level0/` | probes of the recipe, the loss curve, the adapter's weights and the example lengths; the loop runs the first four by default, `lora_weights` and `example_length` when they are passed and their file is there | `lr_range`, `epoch_count`, `lora_rank`, `loss_curve`, `lora_weights`, `example_length` |
| `core/interventions/pairs.py` | fixes and controls that cost at least as much | `make_pair(…)`, `Intervention` |
| `core/interventions/stats.py` | noise floor and the decision of a causal check | `noise_floor(…)`, `causal_check(fix, control, min_effect=…, noise_sd=…)` |
| `core/interventions/tf_runner.py` | a pair as Post-training jobs, adapters scored locally | `TokenFactoryPairRunner` |
| `core/interventions/retrain.py` | the full retrain after approval and the comparison of three models | `plan_retrain(…)`, `Retrainer` |
| `core/connectors/token_factory.py` | Nemotron calls and the Post-training API | `TokenFactory`: `structured`, `chat`, `upload`, `create_job`, `wait`, `events`, `checkpoints`, `download_checkpoint` |
| `core/connectors/storage.py` | the local folder and the Object Storage bucket behind one interface | `Storage`, `get_storage()` |
| `core/connectors/tavily.py` | search and extract with a cache and credit limits | `Tavily.search`, `Tavily.extract` |
| `core/telemetry/calls.py` | the log of model calls and their cost | `CallLog` |
| `jobs/eval_local/exact_match.py` | loads a Token Factory adapter and scores exact match | `evaluate_adapter(…)`, `evaluate_base(…)` |
| `apps/api/` | the HTTP API, the investigation threads and their events, the screen | `main.py`, `runners.py`, `store.py`, `static/index.html` |

`apps/` imports `core/`, and `core/` never imports `apps/`. Inside `core/`, the agent package is the one that knows the others: it imports the planner, the probes, the interventions, the connectors and telemetry. The planner imports nothing else from `core/`, the probes import only the agent's `InvestigationContext`, and the connectors import only the settings. `bench/` and `scripts/` sit on top and import what they need.

## Contracts

```python
# core/probes/base.py
class ProbeResult(BaseModel):
    outcome: str  # one of Probe.outcomes
    raw_ref: str  # storage key with raw data for the UI card
    cost_usd: float = 0.0
    duration_s: float = 0.0
    summary: str = ""


class Probe(ABC):
    name: ClassVar[str]
    level: ClassVar[int]  # 0: CPU and Nano, 1: checkpoint inference, 2: mini fine-tune
    outcomes: ClassVar[tuple[str, ...]]

    def available(self, ctx) -> bool: ...  # a probe without the file it needs says no

    @abstractmethod
    def estimate_cost(self, ctx) -> float: ...

    @abstractmethod
    def run(self, ctx) -> ProbeResult: ...


# core/agent/schemas.py
class Hypothesis(BaseModel):
    id: str
    fault_class: FaultClass  # one of the 12 catalog classes or "other"
    location: str  # data file, config field or code path
    prior: float  # 0 to 1
    rationale: str
    sources: list[str] = []  # URLs from the known issues that support it


class Verdict(BaseModel):
    hypothesis_id: str
    status: Literal["confirmed", "rejected", "inconclusive"]
    evidence: list[str]  # probe and intervention ids
    patch: str  # unified diff for data prep, config or training code
    summary: str = ""  # two or three sentences for the report


# core/case.py
class InvestigationCase(BaseModel):
    task: str
    symptom: str
    mode: Literal["A", "B"] = "A"
    job: FineTuningJob  # id, model, hyperparameters, checkpoints
    datasets: DatasetSplits  # train, validation, check and test: four different files
    evaluator: EvaluatorSpec  # name, metric, higher_is_better, parameters
    budget_usd: float = 5.0
```

The loop hands every probe an `InvestigationContext` (`core/agent/context.py`): the investigation and job IDs, the mode, the job's config, the files by role (`train`, `validation`, `adapter` and others) and the budget left. `InvestigationCase` describes the whole input of an investigation. The loop does not read it yet; it takes a benchmark case as a dictionary.

## API and events

| Method | Path | What it does |
|---|---|---|
| GET | `/` | the investigation screen |
| POST | `/api/investigations` | starts a replay of a saved investigation or a live run: `backend` is `replay`, `local` or `tf`; a replay also takes `demo` and `delay_s` |
| GET | `/api/investigations` | the list of investigations |
| GET | `/api/investigations/{id}` | status, backend and case |
| GET | `/api/investigations/{id}/events` | the events as server-sent events |
| POST | `/api/investigations/{id}/retrain` | approves or declines the proposed full retrain |
| GET | `/api/demo-cases` | the saved investigations that can be replayed |

The event types are `started`, `step`, `awaiting_approval`, `finished` and `failed`. A `step` event carries its kind: `known_issues`, `hypotheses`, `probe`, `intervention`, `verdict`, `retrain_proposal`, `retrain`, `comparison`, `retrain_declined` or `stop`. Events are numbered by `seq` in `events.jsonl`, and each connection streams them from the start, so a reloaded screen redraws the whole investigation.

## Storage

There is no database yet. An investigation lives in the process's memory and in files behind the interface of `core/connectors/storage.py`. `LocalStorage` writes to `./artifacts/`, and `S3Storage` writes to the bucket `next-experiment-artifacts` in Nebius Object Storage, region `eu-north1`, when `STORAGE_BACKEND=s3`. The agent's code does not know which one it has.

```
storage keys: ./artifacts/ locally, the bucket with STORAGE_BACKEND=s3
├── investigations/<id>/
│   ├── meta.json          # status, backend, case, title
│   ├── events.jsonl       # every event the screen got, numbered by seq
│   ├── report.json        # hypotheses, steps, verdict, model calls and cost, retrain plan
│   └── tf/<hypothesis>/   # jobs.json: Post-training job IDs, adapters and their scores
├── tavily-cache/          # Tavily responses keyed by a hash of endpoint and parameters
└── smoke/                 # hello.txt from scripts/smoke_object_storage.py

local files in ./artifacts/, outside the storage interface
├── bench/                 # benchmark rows, case adapters, cached pairs
├── b9/, b10/, b25/        # simulation sweeps, the 200-query check set, weight statistics
└── probes/                # base-model weight norms cached by the weight probe
```

`artifacts/` is not in Git.

## Configuration

Settings come from environment variables through `core/config.py`. `.env.example` lists them without values.

| Group | Variables | Notes |
|---|---|---|
| Token Factory | `TOKEN_FACTORY_API_KEY`, `TOKEN_FACTORY_BASE_URL`, `NEMOTRON_NANO_MODEL`, `NEMOTRON_SUPER_MODEL`, `NEMOTRON_ULTRA_MODEL` | model IDs as the Token Factory console shows them |
| Hugging Face | `HF_TOKEN` | tokenizers and base weights for local scoring and the weight probe |
| Storage | `STORAGE_BACKEND`, `ARTIFACTS_DIR`, `S3_ENDPOINT_URL`, `S3_REGION`, `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | `local` or `s3`; the S3 keys come from the account owner |
| Tavily | `TAVILY_API_KEY`, `TAVILY_MAX_CREDITS_PER_INVESTIGATION`, `TAVILY_CREDIT_BUDGET`, `TAVILY_CACHE_PREFIX` | 10 credits per investigation and 200 per process by default |
| Demo | `PUBLIC_DEMO` | `true` allows only replays |
| Not read yet | `DATABASE_URL`, `DEFAULT_BUDGET_USD`, `APPROVAL_THRESHOLD_USD` | set aside for a database and budget checks |

## Runs and checks

| What | Where | How it runs |
|---|---|---|
| The app | `apps/api/` | `uv run uvicorn apps.api.main:app`; live runs also need `--group ml --group eval` |
| Investigations and the retrain from the command line | `scripts/investigate_tf.py`, `scripts/retrain_tf.py` | the same loop without the API |
| Benchmark | `bench/run_bench.py` | `python -m bench.run_bench --split dev`; `--resume` continues a stopped run |
| Checkpoint pruning | `scripts/tf_prune_checkpoints.py` | keeps the account under Token Factory's limit of 5000 files |
| Public demo | `scripts/build_pages.py`, `.github/workflows/pages.yml` | the screen in static mode and the saved runs, published to GitHub Pages on pushes to `main`; `uv run python -m scripts.build_pages --out _site` builds it locally |
| Checks | `.github/workflows/ci.yml`, `.pre-commit-config.yaml` | GitHub Actions on every pull request and push to `main`: ruff, the format check and pytest; gitleaks before each commit |
| README animations | `docs/animation/` | `uv run --no-project --python 3.14 --with manimgx==0.1.1 python docs/animation/architecture.py`, and `causal_check.py` the same way |
| Storage bucket | created once with the Nebius CLI | Nebius Object Storage, `next-experiment-artifacts` |

## Not built yet

The project plan has parts that the code does not have yet; their folders hold empty placeholders.

- a separate worker process (`apps/worker/`) with a queue and an `events` table in SQLite (`core/db/`), in place of threads and files;
- cancelling an investigation, a report endpoint and a health check;
- approval before paid steps other than the full retrain;
- a screen built with Next.js and a Dockerfile;
- `core/connectors/compute.py`, `core/eval/`, `core/report/` and `jobs/sandbox/`.
