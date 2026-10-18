# Idempotent explicit-case starts

`POST /api/investigations/from-case` accepts an optional `Idempotency-Key` header
containing 1–128 visible ASCII characters. Generate a fresh UUID for each intended
investigation and retain it when retrying that request.

The first accepted request returns `{ "id": "...", "reused": false }`. A repeat
with the same key and validated request body returns the original ID with
`reused: true`, without launching another worker. This includes finished, failed
and interrupted investigations. The same key with a changed case or experiment
returns HTTP 409. Missing keys preserve the previous behavior: each request
creates an investigation. Public-demo restrictions apply before key lookup.

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

The store lock makes concurrent retries safe within the current single-process
API deployment. This is not a distributed lock: multiple API workers or separate
processes sharing storage need transactional request claims. A crash after saving
the claim but before starting its worker leaves an interrupted run; a retry returns
that ID rather than risking duplicate paid work. Recovery of that run is separate
from idempotency. Claims are retained as long as their investigation metadata.
