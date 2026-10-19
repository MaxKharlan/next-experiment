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
