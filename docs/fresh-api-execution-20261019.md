# Fresh API execution — 19 October 2026

PR #89 is merged as `4e374a2`; runtime and execution support from PR #90 is merged
as `9c4f0a0`. The six development jobs have succeeded. The B9 owner confirmed
`MPS_HANDOFF_READY` at 10:07:46 UTC after exit code 0 and validation of all 210
summaries. Development scoring is now running on MPS. No fresh reference training or final-test model evaluation
has run yet.

The development protocol digest is
`32e4a4ef030c323985001327e13274576b3cf724bc2805ef9d9fd464a321acdd`.
Its state lives in `artifacts/fresh-answer-development-20261019/`. The three healthy
jobs report 114,262 trained tokens each; the three damaged jobs report 111,610 each.
At the verified tariff, their total is **$0.2710464 before taxes**, not an invoice.
All three weight digests differ within each arm. Provider model and recipe metadata
match the requested values.

The local runtime and detailed evidence format are described in
[pinned-evaluation-runtime.md](pinned-evaluation-runtime.md). The cohort and
stopping rule remain those in [fresh-answer-preparation.md](fresh-answer-preparation.md).
The [analysis lock](results/fresh-api-execution-20261019/power-analysis-lock.json)
was recorded with zero development scores present. [Inference prices](results/fresh-api-execution-20261019/inference-prices.json)
were checked against the authenticated provider table before API calls.

## Continue the recorded run

Use the same checkout, dependencies, cohort and protocol. Do not regenerate them.
Before resuming, inspect `artifacts/fresh-answer-development-20261019/active-scoring.json`
and its recorded process. The current scoring command runs in exec session `32934`,
PID `82655`; do not start another scorer while it is alive. Its six result files are
written under `scores/` as each adapter completes. If the process exits, inspect its
exit status before running the power gate.

Each paid submission command retains job IDs; a missing ID after an attempted
submission requires provider reconciliation, not a retry.

1. Confirm the B9 process exited successfully and all 210 summaries exist. Obtain
   the owner's explicit `MPS_HANDOFF_READY` and save its evidence in the development
   root's `mps-handoff.json` with `released: true`.
2. Run `scripts.fresh_answer_calibration score` with `--root` and the digest above.
   Run `scripts.fresh_answer_power` with those same arguments after all six scores
   exist. Stop if its decision is not `passed`.
3. Run `scripts.fresh_answer_references prepare`, providing `--development-root`,
   `--cohort artifacts/fresh-extraction-20261019-v1`, a new `--root`, and the
   development digest. Use the newly written reference digest for `submit` and
   `collect`; these commands require an explicit `--env-file` for credentials.
4. After all three references succeed, run `scripts.fresh_answer_api_case` with a
   new `--root`, `--references` and the reference digest. It measures the strong
   original job on check, freezes the request and reserves the API allocation in
   the parent ledger. The final test is not scored.
5. Start `scripts.serve_fresh_api --env-file <credentials-file> --database
   artifacts/fresh-api-20261019.sqlite3 --port 8765`. Validate the saved request,
   then submit it to `/api/investigations/from-case` with the frozen idempotency key.
   Save the returned investigation ID and every approval/report response.
6. At an idle approval boundary, save the report, stop and restart that API process,
   and verify the same proposal survived. Approve only the prespecified answer-drop
   pair, with unchanged data hashes, line-selection hashes, recipe and seed count.
   Decline another paid intervention or a second pair.
7. If a final answer-drop retrain is proposed, inspect its frozen data patch and
   spending reservation before approval. The stopping decision and recipe must be
   fixed before final-test scoring. Preserve an inconclusive outcome without trying
   another seed, recipe or investigation against the same final test.

Save the detailed evaluations, provider metadata, SQLite event export and actual
request/response trace. Reconcile the API ledger with the parent allocation while
retaining any unresolved reservations. Report token-based training estimates
separately from invoices. Describe target/control row and token counts separately;
equal row count is not equal compute. Report control versus the original model
as a descriptive check, and compare the fixed model with the healthy reference
before discussing recovery. One reference per condition does not close a 3×3 study.
