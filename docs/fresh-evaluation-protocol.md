# Fresh evaluation protocol

Status: specification for the next run; no new test set or paid run is claimed.
The historical 13-case benchmark and later inspected cases are development or
exploratory evidence now. Do not label another pass over them as an untouched test.

## Freeze before execution

Record a protocol ID, repository commit, task/model revision, tokenizer/template
hashes, evaluator version and all train/validation/check/test file hashes. Detect
exact overlap and review semantic overlap. Keep the original clean references and
injection line manifest in the evaluator workspace. Neither agent prompts, probes
nor intervention builders may receive them.

Use distinct train, validation, check and final test splits. Inject answer damage
only in train/validation. The agent may inspect observed training data and job
telemetry. It uses check scores for intervention decisions. The evaluator opens
final test only after recipe choice and stopping; further changes require a new
test partition. Data creation and semantic review must precede those hashes.

Freeze T=1, prompt versions, probe/signature versions, allowed interventions and
budget policy. Do not import the experimental all-low-LR classifier. Treat
hypotheses-v4 and verdict-v3 as changed policies requiring its own evaluation, not an improvement
already established by older benchmark scores.

## Reference and control arms

For every claimed task/model cell, record base-model, healthy, damaged, targeted
fix and control scores. Test both mild and strong corruption plus clean inputs.
The target uses observed probe flags only. The control drops an equal number of
unflagged examples. Record retained examples, rendered tokens, epochs, trained
tokens and actual recipe; do not describe this control as exactly cost matched.
If compute imbalance can explain a result, add a prespecified matched-compute
sensitivity comparison before claiming specificity.

Estimate noise separately for each environment/task/model using development
repeats. Choose the minimum useful effect before intervention results. A 0.06
threshold used historically is not a transferable noise calibration. Choose the
seed count from the power analysis; three per arm is only the minimum allowing
an exact one-sided permutation p of 0.05, not a guarantee of adequate power.
Token Factory repeats use independently shuffled training orders because the
recorded provider job seed did not create independent weights.

Report effect, interval, permutation p, seed counts and whether its gate was
applicable. With two seeds, disclose the gate's inability to reach 0.05; do not
present interval-only evidence as a successful permutation test. State the gap
to healthy quality alongside benefit over control.

## Specificity and costs

Test competing explanations on development cases: for example, an unrelated
rank change on a planted low-learning-rate case and answer-drop interventions on
clean prefix labels. Report useful interventions separately from correct cause
identification, healthy false alarms and recovery to the healthy reference.

Before any paid submission, fill in the per-cell job count, conservative token
estimate, inference reserve, retrain estimate, total cap and approval owner.
Save the approved protocol hash in the run. Stop on cap exhaustion, ambiguous
provider state or protocol/file changes. Retain uncertain reservations until
reconciled. A missing cap or unreviewed data blocks submission, not offline
preparation.

Close a matrix cell only with linked job/run IDs, input/model/evaluator hashes,
per-seed metrics, cost records and the final untouched-test outcome. Close the
end-to-end demo only with an actual persisted API trace, including hypotheses,
probe observations, approval and target/control execution. A standalone
experiment replay does not substitute for that trace.
