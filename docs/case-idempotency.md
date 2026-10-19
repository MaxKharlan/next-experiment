# Idempotent explicit-case starts

`POST /api/investigations/from-case` accepts an `Idempotency-Key` header
containing 1–128 visible ASCII characters. Generate a fresh UUID for each intended
investigation and retain it when retrying that request. SQLite-backed API workers
require the header for explicit cases and live `POST /api/investigations` starts.
It remains optional on the legacy single-process storage backend.

The first accepted request returns `{ "id": "...", "reused": false }`. A repeat
with the same key and validated request body returns the original ID with
`reused: true`, without launching another worker. This includes finished, failed
and interrupted investigations. The same key with a changed case or experiment
returns HTTP 409. Missing keys return 422 with SQLite state; on the legacy storage
backend they create independent investigations. Public-demo restrictions apply
before key lookup.

Keys are stored as hashes alongside a hash of the normalized request body in the
investigation metadata. Key matching therefore survives a restart. A repeated
request resolves before local dataset validation: it retrieves the original run,
even if those local files were later removed. It does not rerun an old investigation
against changed files. To intentionally start another investigation, use a new key.

The screen generates a key for each start attempt. If the response is lost, it
retains that key for retries of the unchanged JSON in the same page. Revalidate
and retry without editing the request. A successful response clears the key so a
later deliberate start can create a new run. The key is not retained across page
reloads; check Past runs before starting again after a reload.

SQLite request claims are serialized in a database transaction, including across
API processes. The legacy storage backend retains a process-local lock and rejects
a second API process using the same artifacts directory on the same host. See
[worker ownership and deployment](sqlite-investigations.md).

A crash after saving the claim but before dispatch leaves the original run queued
until its lease expires. A retry returns that ID without starting another worker.
Expiry records failure; it does not take over or repeat the execution. Claims remain
as long as their investigation metadata. A new key expresses a new intended run;
it does not reconcile a previous unknown provider outcome.
