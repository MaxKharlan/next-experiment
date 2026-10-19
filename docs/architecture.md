# Architecture

Next Experiment serves a static investigation screen from a Python HTTP API.
Each investigation runs in a background thread owned by an API worker. Nemotron
on Token Factory supplies model responses; Token Factory trains LoRA adapters,
which the app downloads and evaluates locally. GitHub Pages serves a replay-only
build of the same screen.

The default `storage` state backend permits one API process. The `sqlite` backend
supports multiple API workers on one host with a shared database, working directory
and artifact storage. It uses execution leases and atomic request/approval claims.
There is no separate worker service or automatic takeover of interrupted paid work.
See [worker coordination and migration](sqlite-investigations.md).

## The path of an investigation

1. The API validates a live request and its four data splits before dispatch.
   Explicit cases enter through `/api/investigations/from-case`; the registered
   demo case and replays use `/api/investigations`. SQLite live starts require an
   `Idempotency-Key`. Repeated matching requests return the original investigation.
2. A background thread changes `queued` to `running` and writes events through
   the investigation store. Token Factory calls and uncached Tavily calls check
   the worker's execution ownership before sending.
3. Tavily searches for known issues using model/recipe metadata. The router sends
   hypotheses and probe interpretation to Super; the planner selects probes using
   the outcome likelihood table and expected information gain per dollar.
4. Paid probes and causal checks pause at `awaiting_paid_step`. The API saves the
   continuation and proposed cost. Approval includes that event's `proposal_seq`;
   an already consumed or stale approval cannot launch another execution.
5. The approved check builds target/control arms, trains the configured seeds and
   evaluates on the check split. It records examples, training tokens and costs
   for both arms. Equal row counts need not mean equal training costs.
6. The statistical rule sets `confirmed`, `rejected` or `inconclusive`. A confirmed
   result supports the tested intervention against its control; it does not alone
   identify a unique cause or establish full recovery. Ultra writes the summary;
   the code supplies the decision, evidence and patch.
7. A supported fix can produce a final retrain proposal at `awaiting_approval`.
   Approval through `/retrain`, again with `proposal_seq`, trains the fixed recipe
   or repaired data and compares base, failed and fixed models on the test split.
   Dataset hashes are checked before paid training and retraining.
8. The run becomes `finished`, or `failed` with available events and spending
   evidence. Finished runs and waiting approvals survive restart. An interrupted
   storage-backed run fails on reload; a SQLite run fails after its lease expires.
   Neither condition automatically restarts provider requests.

The [live protocol](live-protocol.md) defines split use and budget semantics.
The local backend and historical replay records have narrower contracts; they do
not establish that a new Token Factory investigation passed those checks.

## Model routing and evidence

| Step | Route | Use |
|---|---|---|
| `hypotheses`, `interpret_probe` | Super; Ultra after two mismatches | Main agent loop |
| `verdict` | Ultra | Summary of the code's intervention decision |
| `probe_code` | Super | Opt-in generated local checks through the Python entry point |
| `react`, `single_shot` | Super, Ultra respectively | Benchmark baselines |
| `audit_example`, `label_errors`, `parse_events` | Nano | Reserved mappings; no automatic per-example Nano pipeline |
| `known_issues`, `patch` | Super, Ultra respectively | Reserved mappings; ordinary search queries and recipe patches are built in code |

Prompts live in `core/agent/prompts/`; current verdict wording is `verdict-v3`.
Call logs record each completed structured-output attempt before JSON validation
or correction. Usage events include available provider request and response IDs
and the local spending operation ID. A response that fails usage validation or an
HTTP error can still leave a provider receipt in the spending ledger.
See [persisted usage and receipts](persisted-llm-usage.md).

Signature probabilities are development estimates or heuristics within their
recorded coverage. Temperature remains T=1; the experimental B9 classifiers stay
offline after failed provider transfer. Saved historical heldout results have
been reused for analysis; a fresh final test requires new, unviewed cases.

## Modules

| Path | Responsibility |
|---|---|
| `apps/api/main.py`, `runners.py`, `store.py` | Request validation, thread dispatch, ownership, persistence and events |
| `apps/api/static/index.html` | Live/replay screen, paid decisions, costs and report downloads |
| `core/case.py` | Typed case and data split contracts |
| `core/agent/` | Investigation context and loop, model routing, schemas and versioned prompts |
| `core/planner/` | Information gain, stopping rules, outcome likelihoods and training cost estimates |
| `core/probes/` | Recipe, loss, length, answer-damage, adapter and supplied-label checks |
| `core/interventions/` | Target/control builders, statistical decisions, training runners and retraining |
| `core/connectors/` | Token Factory, Tavily and artifact storage |
| `core/db/` | Durable training/inference reservations and Tavily credit reservations |
| `core/telemetry/` | Per-attempt call log, model usage and provider references |
| `core/eval/`, `jobs/eval_local/` | Dataset audits, checkpoint loading and local quality evaluation |
| `core/report/` | Investigation reports and provider-cost reconciliation |
| `jobs/sandbox/` | Opt-in bounded local Python execution |
| `bench/`, `scripts/` | Benchmarks, development experiments, exports and operational tools |

Mode-B supplied-label audits are available in REST/UI and through the Python
entry point. Local code-switch interventions and synthetic CPU quality checks
exist, but arbitrary-code repair and a complete pretrained mode-B investigation
remain unvalidated. See [the mode-B package](mode-b-local-package.md).

## API and events

| Method | Path | Contract |
|---|---|---|
| GET | `/`, `/api/capabilities`, `/api/demo-cases` | Screen, available features and saved cases |
| POST | `/api/investigations` | Start replay, registered local or Token Factory investigation |
| POST | `/api/investigations/validate` | Validate an explicit case before launch |
| POST | `/api/investigations/from-case` | Start a validated explicit Token Factory case |
| POST | `/api/investigations/label-audit`, `/api/investigations/audit-case` | Persist local label or dataset audits |
| GET | `/api/investigations`, `/api/investigations/{id}` | List and inspect state |
| GET | `/api/investigations/{id}/events` | Persisted server-sent events |
| GET | `/api/investigations/{id}/report` | Report JSON; `?download=true` adds attachment headers |
| POST | `/api/investigations/{id}/paid-step` | Approve/decline the current paid probe or check |
| POST | `/api/investigations/{id}/retrain` | Approve/decline the current final retrain |

Events include `started`, `step`, `llm_call`, `awaiting_paid_step`,
`awaiting_approval`, `finished` and `failed`. Their persisted sequence numbers are
SSE IDs. A fresh connection can replay the full history; `Last-Event-ID` resumes
after the last received event. SQLite readers reload state written by other
workers. The report exposes steps, usage and available spending snapshots.

## Storage and paid requests

`INVESTIGATION_BACKEND` selects the state store independently of `STORAGE_BACKEND`,
which selects local or S3 artifact storage. SQLite state uses
`investigation_snapshots` and `investigation_leases`. Storage-backed state uses
metadata and event files. Switching backends does not import existing history.

Live paid runners also use local files under `./artifacts/investigations/<id>/`,
including `spending.sqlite3`, submission journals, downloaded adapters and scores.
Workers must therefore share their working directory even with S3 artifacts.
`artifacts/` is ignored by Git; selected reports are exported to `docs/results/`.

Before inference or training, the spending ledger reserves the estimate. Known
usage settles the reservation; ambiguous outcomes keep it pending. The separate
`provider_receipts` table links returned request/response IDs to reservations,
including HTTP errors. Provider receipts do not settle a hold or establish a billed
amount. Training submission journals retain known job IDs and block blind resubmission
after an unknown outcome. Tavily has its own durable credit ledger.

[Paid-operation recovery](paid-operation-recovery.md) describes operator checks.
The leases and local journals cannot make a remote service exactly-once. Provider
invoice reconciliation remains necessary when responses or billing details are
missing.

## Configuration and operation

Settings come from `core/config.py` and `.env`; `.env.example` lists the variables.

| Group | Variables and behavior |
|---|---|
| Models | `TOKEN_FACTORY_API_KEY`, `TOKEN_FACTORY_BASE_URL`, `NEMOTRON_*_MODEL`; `HF_TOKEN` for local model access |
| State | `INVESTIGATION_BACKEND=storage` by default; `DATABASE_URL` is used with `sqlite` |
| Artifacts | `STORAGE_BACKEND`, `ARTIFACTS_DIR`, `S3_*`; paid runner working directories must also match |
| Tavily | `TAVILY_API_KEY`, `TAVILY_MAX_CREDITS_PER_INVESTIGATION`, `TAVILY_CREDIT_BUDGET`, `TAVILY_CACHE_PREFIX`, `TAVILY_LEDGER_PATH` |
| Runtime | `PUBLIC_DEMO=true` allows replays only; `HYPOTHESIS_SAMPLES` controls hypothesis sampling |
| Reserved settings | `DEFAULT_BUDGET_USD` and `APPROVAL_THRESHOLD_USD` are declared but do not govern live approval/budget decisions; live budgets come from the validated protocol |

Use the [README run commands](../README.md#run-it) and the
[SQLite deployment instructions](sqlite-investigations.md). The public site is
built by `scripts/build_pages.py` and deployed by `.github/workflows/pages.yml`
on pushes to main. CI runs Ruff and pytest; browser-script regression tests use
Node, while layout changes also need visual checks in a browser.

Cancellation, a dedicated health endpoint, a cloud compute connector, a separate worker queue, multi-host
coordination, a Next.js frontend and hosted Spaces deployment remain open. Current
experimental acceptance criteria are tracked in [Project status](project-status.md).
