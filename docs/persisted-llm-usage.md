# Persisted LLM usage

Live API investigations now write an `llm_call` event whenever the router adds a
response to its call log. These events use the investigation store, including its
configured persistence backend. They remain available after a failed worker or an
application restart.

The report endpoint exposes `llm_usage` with `call_count`, `tokens_in`,
`tokens_out`, `cost_usd` and individual `calls`. Each call records its model,
purpose, prompt version, attempt number, latency and timestamp. Prompts and model
response text are not stored in these usage events.

The Token Factory connector notifies the router immediately after each completed
structured-output response, before validating JSON or issuing a correction
request. The router records that attempt and the live API persists its usage
event. A later transport error therefore leaves earlier completed attempts in
the report. Returning or raising the accumulated attempts does not log them a
second time. Adapters accepting `on_call` but returning only a final call list
still use the router's fallback logging.

If the persistence callback fails, the exception stops the request sequence;
no correction request is sent after that failure. This does not guarantee a
record when storage itself is unavailable. A crash before usage is received,
or a provider request whose response is lost, can still leave an unrecorded
charge. The log is not a provider invoice or complete billing reconciliation.

`llm_usage` is a separate view of recorded model calls. Do not add it to the
summary LLM cost, which covers the same calls on a completed run. It excludes
training and Tavily charges. Existing reports without usage events return null,
including zero-call runs; null does not assert that spending was zero. Existing
summary fields are preserved.

The SSE stream includes the additional `llm_call` event type. Consumers can use it
for usage displays or ignore it. Investigation status and causal step ordering
are unchanged. This supports B17 cost accounting; it does not implement a hard
spending cap, a durable worker queue, or automatic recovery of in-flight jobs.

## Provider references and budget reservations

Each new call has three nullable fields:

| Field | Source and purpose |
|---|---|
| `provider_request_id` | SDK response `_request_id`, populated from the provider's request-ID header |
| `provider_response_id` | Completion response `id`; a separate reference from the HTTP request ID |
| `spending_operation_id` | Local reservation ID linking this call to its spending ledger; null when no ledger is attached |

The same fields survive the call JSONL, `llm_call` event, resumed call log and
report export. Each structured-output correction has its own references and
reservation. Older records and responses without identifiers retain null values;
no IDs are reconstructed from timestamps or costs.

Before validating usage or settling a reservation, the connector saves a receipt
in `provider_receipts` in the investigation's `spending.sqlite3`. The receipt
records the reservation ID, available provider IDs, outcome (`response` or
`http_error`), HTTP error status when present and receipt time. HTTP errors use
the SDK exception's request ID. A response without usage can therefore leave
identifiable evidence while its reservation stays pending. Spending snapshots
include these receipts, including the snapshot attached to a failed run.

Recording the same receipt twice is harmless; replacing it with conflicting
references fails. A receipt write failure stops the request sequence before a
structured-output retry. No headers, credentials, prompts or response text are
stored in this table. If transport fails before any provider identifier arrives,
the pending reservation remains the only local evidence. Operators still need
the provider's records to establish whether it was charged.

Use these identifiers when matching provider request exports or support records.
They improve traceability for new calls; they do not backfill historical calls,
represent provider idempotency keys or prove invoice reconciliation. See the
[current reconciliation record](provider-cost-reconciliation-20261019.md).

## Investigation screen

The cost card updates as `llm_call` events arrive, grouping calls and tokens by
model. Retry attempts have separate call IDs and are included. Repeated delivery
of the same call ID is ignored. Selecting another investigation clears the local
usage journal; replaying its stored events rebuilds the card.

Costs remain visible if the investigation fails. The final summary is not added
to the recorded calls a second time. Saved demos without per-call events still
use their original summary. The card distinguishes estimated training costs and
states that pending requests and Tavily charges are excluded. It shows recorded
usage, not a live provider invoice or a spending limit.
