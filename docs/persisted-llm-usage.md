# Persisted LLM usage

Live API investigations now write an `llm_call` event whenever the router adds a
response to its call log. These events use the investigation store, including its
configured persistence backend. They remain available after a failed worker or an
application restart.

The report endpoint exposes `llm_usage` with `call_count`, `tokens_in`,
`tokens_out`, `cost_usd` and individual `calls`. Each call records its model,
purpose, prompt version, attempt number, latency and timestamp. Prompts and model
response text are not stored in these usage events.

The router logs the attempts returned by a successful structured request and the
attempts carried by `StructuredOutputError`. Thus a failed JSON validation can
still contribute recorded costs. Persistence occurs when the router receives
those results; a crash while the provider is still working, or a transport error
without returned usage, may leave an unrecorded charge. This is not a provider
invoice or a complete billing reconciliation.

`llm_usage` is a separate view of recorded model calls. Do not add it to the
summary LLM cost, which covers the same calls on a completed run. It excludes
training and Tavily charges. Existing reports without usage events return null,
including zero-call runs; null does not assert that spending was zero. Existing
summary fields are preserved.

The SSE stream includes the additional `llm_call` event type. Consumers can use it
for usage displays or ignore it. Investigation status and causal step ordering
are unchanged. This supports B17 cost accounting; it does not implement a hard
spending cap, a durable worker queue, or automatic recovery of in-flight jobs.

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
