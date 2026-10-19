# Approved answer-drop retraining

A confirmed answer-drop check previously ended without a retrain proposal. The
API now proposes one fixed-model job using the exact training-line deletions
selected by the check. The original failed job remains the comparison model.
The original recipe and validation file remain unchanged.

The existing retrain approval endpoint accepts or declines this proposal. Its
saved plan contains all four dataset hashes, the source training hash, deleted
line numbers, retained row count and the materialized training hash. Approval
survives an API store restart. A declined proposal submits no training job.

Before submission, the executor reconstructs the plan from the frozen protocol
and rejects changed files, recipe, estimate or row selection. It writes a separate
`fixed-train.jsonl` under the investigation's retrain directory. It never edits
the source dataset. Final scoring of base, original failed and fixed models uses
only the protocol's final test; validation is not a fallback. Files are checked
again before scoring and after the comparison. The report includes the patch,
model, recipe, evaluator and dataset hashes under `answer-data-retrain-v1`
provenance.

The estimate uses the original source row count and configured mean tokens per
row before deletion. This avoids understating retained token counts when shorter
rows are removed, but still depends on the supplied token estimate and price
model. The retained row count is reported separately. The investigation's SQLite
ledger reserves this training estimate before uploads or job creation. Settlement
records the same estimate, not a provider invoice. The final summary includes it
in training and total estimates.

A completed result is reused without more jobs or scoring. An interrupted attempt
leaves its reservation pending and refuses automatic retry, including when the
provider may have accepted a job without returning its ID. Reconcile provider
state and the reservation before recovery; no automatic recovery command is
provided. The generic retrainer rejects data patches so a legacy caller cannot
silently train the unmodified source file.

## Validation and remaining work

Provider-stub API tests cover confirmation, restart, approval/decline, exact row
deletions, separate final-test scoring, report download and budget accounting.
Unit tests cover changed splits, a changed estimate, an incorrect scoring split,
insufficient funds, an ambiguous submission and data changes during training.
The full suite passes 741 tests with 15 skips; Ruff and formatting pass.

No paid job or real model evaluation was run for this change. The fresh dataset,
review, frozen protocol, measured full API trace and healthy-quality comparison
remain pending under the [fresh evaluation protocol](fresh-evaluation-protocol.md).
This implementation does not close general multiworker ownership, invoice
reconciliation or the legacy recipe-retrain budget gap. B9's active sweep and
subsequent transfer work remain owned by its existing chat.
