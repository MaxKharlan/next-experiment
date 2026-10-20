# REST investigations for supported trainer faults

`POST /api/investigations/code` starts a mode-B investigation for one of four
explicit trainer switches: `prompt_loss`, `eos_as_pad`, `double_shift` or
`microbatch_mean`. The request names disjoint train/validation/check files, a pinned
Qwen3-0.6B runtime, recipe, seeds, device and inference budget. An
`Idempotency-Key` is required. Public-demo mode rejects the endpoint.

The worker asks Super for hypotheses and tests at most one supported repair
against the unchanged switch. Each arm trains fresh adapters. The existing
intervention rule supplies the decision; Ultra is called only for a supported
verdict. Three default seeds are exploratory; a request may provide up to six.
The route does not execute arbitrary user Python.

Configure `NEXT_EXPERIMENT_SPENDING_LEDGER` and
`NEXT_EXPERIMENT_SPENDING_CAP_USD` for the authorized global allocation. A second
ledger enforces the request's own limit. Both reserve model-call costs with a 20%
allowance. Unknown outcomes retain holds, and the provider client makes no
automatic paid retries. Local training time is not included in provider charges.

Before training, validation rejects split overlap, context truncation and changed
model/data/source hashes. A process lock serializes these API workers' model
execution. Separate command-line model processes must be coordinated by the
operator; they do not acquire this lock.

Accepted requests and events use the existing durable store. Read progress at
`/api/investigations/{id}/events` and the report at
`/api/investigations/{id}/report`. Reposting the same key and body after a restart
returns the original ID; a changed body is rejected. Interrupted training is
reported as failed and is not automatically resumed.

The prospective [trace protocol](results/rest-code-investigation-20261020/protocol.json)
uses four seeds per arm and preserves inconclusive outcomes. Route tests verify
request identity across store reconstruction, budget rejection before dispatch,
source drift and model-lock behavior. A completed unit test is not a live paid
trace; see that protocol's execution artifacts for its current state.
