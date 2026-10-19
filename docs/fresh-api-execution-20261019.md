# Fresh API execution — 19 October 2026

The run stopped at its prespecified development gate with **`zero_sd`**. All six
adapters scored 102/102 (`json_fields_exact = 1.0`) on the historical development
check. No fresh reference training, fresh final-test evaluation or full persisted
API investigation followed. This is a completed stopping decision, not a completed
full API trace or evidence that answer corruption is harmless.

| Development arm | Seed 1101 | Seed 1102 | Seed 1103 |
| --- | ---: | ---: | ---: |
| Healthy | 1.0 | 1.0 | 1.0 |
| Damaged | 1.0 | 1.0 | 1.0 |

The pooled seed-level SD is zero. The frozen rule stops before power simulation
in this case, so no pair seeds were selected and no power estimate was produced.
All three weight digests differ within each arm: this result is not caused by
reusing the same checkpoint. The development check reached its ceiling for both
arms and could not distinguish them. Six perfect scores do not establish zero
population variance or performance on the unscored fresh cohort.

## Evidence and cost

The [power decision](results/fresh-api-execution-20261019/power-decision.json),
[status and spending](results/fresh-api-execution-20261019/status.json), and
[six per-example score files](results/fresh-api-execution-20261019/scores/) retain
job IDs, predictions, expected answers, dataset hashes, adapter hashes and runtime
metadata. All 612 stored predictions were rescored with the frozen metric and
matched the recorded results. The scoring process and power analysis exited with
code 0. No evaluator or stopping-rule changes were made after observing scores.

Three healthy jobs report 114,262 trained tokens each; three damaged jobs report
111,610 each. At $0.40 per million tokens, total estimated training cost is
**$0.2710464 before taxes**, not a provider invoice. All six reservations are
settled; no unresolved holds remain. The $5 ledger retains $4.7289536 available.
Verified [inference pricing](results/fresh-api-execution-20261019/inference-prices.json)
was recorded before execution. No new paid jobs were launched after scoring.

The development protocol digest is
`32e4a4ef030c323985001327e13274576b3cf724bc2805ef9d9fd464a321acdd`.
The [protocol](results/fresh-api-execution-20261019/protocol.json) and
[analysis lock](results/fresh-api-execution-20261019/power-analysis-lock.json)
were fixed with zero scores present. Local working state remains in
`artifacts/fresh-answer-development-20261019/`; its `completion.json` records the
terminal decision. Do not rerun selection or submit the prepared fresh jobs under
this stopped protocol.

## Integration and resources

PR #89 merged as `4e374a2`; runtime and execution support from PR #90 merged as
`9c4f0a0`. Their implementation validation passed: 813 tests with ML dependencies
(8 skipped), and CI 767 tests (15 skipped). These are software checks, not evidence
of a completed live investigation. Real loopback HTTP preflight returned 200 for
capabilities and an empty investigation list; the paid approval/restart trace was
not reached.

B9 confirmed `MPS_HANDOFF_READY` at 10:07:46 UTC after exit code 0 and validation
of all 210 sweep summaries. The [handoff evidence](results/fresh-api-execution-20261019/mps-handoff.json)
is saved. This task scored the six existing adapters in one MPS process and did
not duplicate B9 training. MPS was released back after scoring, and the recurring
continuation check was disabled after the stopping decision.

The local base snapshot, tokenizer/template, evaluator and package versions were
pinned as described in [pinned-evaluation-runtime.md](pinned-evaluation-runtime.md).
Token Factory does not expose an immutable base-weight revision in these job
records; the local pin does not establish provider/local base-weight identity.
The fresh cohort retains the preparation boundaries in
[fresh-answer-preparation.md](fresh-answer-preparation.md), including a synthetic
bounded task and no independent human review.

A subsequent experiment would need a separately specified development design
that can measure variation and sensitivity, with a new protocol fixed before its
results. This run does not authorize adapting seeds, thresholds or recipes to
bypass the stopping rule, and its untouched fresh final test remains protected.
