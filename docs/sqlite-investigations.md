# SQLite investigation state and API workers (B22/B17)

Use `INVESTIGATION_BACKEND=sqlite` for multiple API workers on one host. All workers
must share the same database file, working directory and artifact storage. Start
from the repository root:

```bash
INVESTIGATION_BACKEND=sqlite \
DATABASE_URL=sqlite:///./artifacts/next_experiment.db \
uv run uvicorn apps.api.main:app --workers 2
```

SQLite stores metadata and event history together in `investigation_snapshots`.
The `investigation_leases` table stores each execution's token, expiry and dispatch
state. Request deduplication, approval consumption and event writes use
`BEGIN IMMEDIATE` transactions. Reads reload the database, so a worker serving
status, reports or SSE sees events written by another worker. A reader does not
mark an active owner as interrupted merely because its own cache is empty.

## Ownership and failure recovery

Creation or approval claims an execution for 60 seconds. Dispatch consumes that
claim once, and a heartbeat renews it every 20 seconds while the runner is alive.
A waiting or terminal event releases ownership in the same transaction that saves
the event. Each execution retains its own token; an older runner cannot acquire the
authority of a later approval through a mutable store reference.

Event writes require the current, unexpired token. Token Factory training and
inference, and uncached Tavily calls, check ownership before sending. Inference
and search check it again after reserving budget. A failed heartbeat blocks further
paid calls from that worker. Existing spending and submission records remain in
place when ownership is lost.

Once a lease expires, the next state read records one `failed` event with reason
`lease_expired`. Expiry never transfers execution to another worker, replays an
approval or restarts a provider request. A repeated start with the same idempotency
key returns the original investigation, including a failed one. Check provider jobs,
request evidence and ledger reservations before planning any replacement work.
An accepted or in-flight provider request cannot be cancelled or made exactly-once
by this local lease; ambiguous outcomes still require reconciliation.

A crash after recording approval but before thread dispatch follows the same rule:
the approval stays consumed and the run expires. The operator can inspect the saved
proposal, decision, partial events and spending records. There is no automatic
paid-run takeover or reconciliation endpoint.

## Request and approval contracts

With SQLite state, both live `POST /api/investigations` and
`POST /api/investigations/from-case` require `Idempotency-Key`. Replays and local
non-model audits do not. Use one key for one intended run and retain it across
retries. A changed body with an existing key returns 409. See
[request identity behavior](case-idempotency.md).

Both approval endpoints require `proposal_seq`, the integer sequence of the waiting
event, on either state backend. For example, if `awaiting_paid_step` has SSE ID 12:

```json
{"approve": true, "proposal_seq": 12}
```

A missing sequence or a sequence from an earlier proposal returns 409. This prevents
a delayed retry from approving the next paid step. The web UI captures the SSE ID in
each decision handler and sends it with the answer. A rejected duplicate decision
does not start a worker or mutate the current proposal.

## Deployment and migration

Stop all old API processes before deploying this version. Do not mix old snapshot
writers with workers that enforce leases. Tables are created at startup under a
write transaction. Existing completed and waiting SQLite snapshots remain readable;
a waiting proposal receives a lease when answered. An old queued/running snapshot
without a lease is recorded as interrupted (`process_restart`) when read, without
restarting its provider work.

`INVESTIGATION_BACKEND=storage` remains the default. Its state locks are local to one
process; the API factory now holds an OS lock at
`<ARTIFACTS_DIR>/.api-single-worker.lock` and rejects a second process on the same
host using that directory. This guard uses POSIX file locking, as supported by the
project's Linux/macOS runtime. It is not a distributed S3 lock. Use SQLite state for
multiple workers, even when artifact storage uses S3. Switching state backends
selects separate history; no file/S3 investigations are automatically imported.

This deployment supports a shared local SQLite file on one host, not independently
mounted database copies, multiple hosts or a network filesystem. Paid runners still
use the repository's `./artifacts` directory, so worker working directories must
match. Snapshot writes rewrite the full event history and serialize with other
writers; a high-throughput queue or database server would require a separate design.
Protect the database and artifacts as private investigation data.

Tests race three spawned processes against the same request key and both approval
kinds, check that only one simulated paid operation occurs, kill an owner with an
unresolved reservation, and verify stale-token rejection and heartbeat renewal.
Provider-boundary tests reject inference, training and search after lease loss.
All provider calls in these tests are simulated.

Validation on 19 October 2026: 861 tests passed, 8 skipped; Ruff lint and formatting
passed. A local Uvicorn server with two workers completed 12 concurrent replay
requests with continuous SSE sequences. The browser script passed Node syntax
checking. These checks made no paid requests.
