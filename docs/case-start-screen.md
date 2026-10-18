# Start an explicit case from the screen

On the local API page, open **Investigate your own job**. Load a JSON request or
paste one containing the `case` and `experiment` objects accepted by
`POST /api/investigations/from-case`. The supported task and model are currently
BANKING77 and Qwen3-0.6B on Token Factory.

Dataset paths refer to files on the API server. Loading the request file in the
browser does not upload training data. The request format and server-side
validation are described in [the live protocol](live-protocol.md).

First select **Validate without training**. Validation checks the local datasets
and settings and shows the job ID, model, example counts, investigation budget
and estimated training cost per causal pair. It makes no provider calls and does
not verify the remote job status. Editing or replacing the request invalidates
this result and disables starting until validation succeeds again.

**Start paid investigation** submits the validated request. The server validates
it again, checks the provider job in the worker and streams the investigation.
The form disables duplicate clicks while a request is pending. A failed start
requires validation again. After a network error, revalidate and retry the unchanged request in the same
page: its retained idempotency key retrieves the original investigation. After
reloading the page, inspect Past runs before starting again. See
[idempotent starts](case-idempotency.md) for persistence and concurrency limits.

The public API demo and static GitHub Pages demo hide this form. The API's existing
public-demo restrictions still reject explicit validation and training requests.
`GET /api/capabilities` reports whether explicit-case controls should be shown.
