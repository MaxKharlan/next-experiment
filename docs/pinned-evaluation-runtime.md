# Pinned runtime for fresh evaluations

Explicit API cases can include `experiment.evaluation_runtime`. It records the
local model revision path, hashes of all seven files in the supported single-file
Qwen snapshot, the chat template and exact evaluator dependency versions. Both
paired checks and approved final retraining pass this binding to the evaluator.

The evaluator validates the files and versions before loading, then loads from the
local snapshot path. A checkpoint template must match the frozen template. The
same checks run after generation. Dataset and adapter file hashes are recorded in
each detailed evaluation result; a change during scoring rejects that result.
Existing cases without a runtime binding retain their previous behavior.

`core.eval.runtime.freeze_runtime` creates the binding without importing model
weights into torch. It hashes weights by streaming bytes. The protocol's local
revision does not establish which immutable base revision Token Factory trained;
the provider request names a model ID. Retain provider job metadata and downloaded
adapter hashes and report that limitation.

`scripts.fresh_answer_calibration` prepares six development-only jobs on the
historical structured cohort, independently shuffling three healthy and three
30%-damaged training files. Preparation freezes data, recipe, code, evaluation
runtime and the stopping/power rules. The authenticated provider price table was
checked on 19 October: Qwen3-0.6B LoRA 8K costs $0.40/million tokens before taxes.
The complete experiment retains a $5 cap; this calibration shares that allocation.

Submission requires the protocol digest. Each reservation and submission attempt
is durable before provider calls; an ambiguous attempt blocks automatic retries.
Successful submissions are reused. Collection saves job metadata, adapters and
trained-token cost estimates, which remain distinct from invoices. Scoring needs
a recorded B9 MPS handoff. It uses only the historical check split and does not
read the new cohort. The new final test remains reserved until the full API
investigation stops and the final recipe is fixed.

Pinned API checks retain the complete `EvalResult` under
`investigations/<id>/evaluations/check/`; final comparisons retain base, failed and
fixed results under `evaluations/final/`. Diagnostic probe contexts receive only
train and validation paths. Check/test validation and scoring remain outside that
context.

The SDK does not retry paid fine-tuning submissions after a transport or server
error. Such an error can follow an accepted job; its reservation must stay held
until the provider state is reconciled.

The remaining execution commands are deliberately separate:

- `scripts.fresh_answer_power` applies the frozen rule once to all six scores. It
  stops on repeated weight digests within an arm, zero pooled SD, or failure of
  every planned candidate. It selects the first passing count and never adds seeds.
- `scripts.fresh_answer_references` prepares and submits exactly one healthy,
  mild and strong fresh reference, using the same parent spending ledger. Its
  seeds are fixed at 4101, 4102 and 4103 before fresh results are observed.
- `scripts.fresh_answer_api_case` scores the strong original job on check and
  freezes the API request, source hashes and budget allocation. It uses the
  observed score in the symptom; a high score can lead to no intervention. It
  leaves $0.50 unallocated for contingency.
- `scripts.serve_fresh_api` starts one loopback worker with SQLite persistence,
  explicit credentials, repository model routing, and weight probes/web search
  disabled. The paid-step and final-retrain endpoints retain their approval gates.

The fresh trace permits one answer-drop target/control pair and one final retrain.
Decline a different paid intervention or a second pair. Restart only while the API
is idle at a persisted approval boundary, then inspect the recovered proposal before
approving it. A stopped or inconclusive run is a valid reported outcome, not a
reason to open a replacement investigation on the same test.
