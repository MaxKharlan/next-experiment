# Three-task reference matrix and noise coverage

The original B5 names Qwen3-0.6B, Qwen3-1.7B and Llama-3.2-1B-Instruct for all
three tasks. A13 supplies evaluation and error breakdowns. A completed cost-fit
job is not a healthy quality reference. The 12/12 Super smoke is also outside
this matrix. The table below inventories published evidence without recomputing
predictions or reading historical final-test data.

| Task | Qwen3-0.6B | Qwen3-1.7B | Llama-3.2-1B-Instruct |
| --- | --- | --- | --- |
| BANKING77 | New clean reference: base 0/154, selected 104/154 | New clean reference: base 1/154, selected 115/154 | Full quality row missing |
| JSON extraction | New clean reference: base 0/102, selected 102/102 | New clean reference: base and selected 102/102 | Full quality row missing |
| Function calling | New clean reference: base 0/102, selected 100/102 | New clean reference: base and selected 102/102 | Full quality row missing |

The new [Qwen reference package](qwen-reference-quality-20261020.md) verifies
5,012 predictions and validation selection. The structured tasks are previously
inspected development splits; every 0.6B base answer has a Markdown fence and
fails the strict JSON contract. The 1.7B base already reaches 100% on those
tasks. Damaged/fixed/control comparisons and B10 noise measurements remain
missing from these six cells.

Sources: [historical benchmark](benchmark.md), [answer-cut studies](answer-cut.md),
[structured tasks](structured-tasks.md), [rejected development series](format-content-development-20261019.md),
[cost-fit scope](model-cost-validation.md). Scores on different datasets or
splits must not be combined into one baseline/healthy/damaged/fixed row.
The [original machine-readable inventory](results/matrix-plan-20261020/plan.json)
is preserved as the prospective planning snapshot. Its null metrics describe
that earlier state; the linked Qwen package supplies the new clean-reference
measurements without filling unrelated fault-comparison gaps.

## Required evidence for each cell

Record base-model quality, a validation-selected healthy recipe, the damaged
recipe, the proposed fix and its matched control. Each score needs the same task
version, clean check IDs, model/tokenizer revisions, decoding, dataset hashes,
seed and prediction artifact. Retain invalid answers and generation-limit hits.
BANKING77 uses label exact match; JSON uses strict full-object exact and schema
validity; function calling uses exact function and arguments. A13 also requires
length/source/type breakdowns, EOS behavior and checkpoint comparisons.

Reference selection follows the existing four-point learning-rate/epoch grid
(1e-4 or 2e-4; three or six epochs). Select on validation, break ties in frozen
grid order, then evaluate the selected recipe once on check. Do not choose a
repair using a final test. The new ticket-ledger preparation is a development
candidate, not a replacement for every A2/A3 reference task or an admitted design.

First admit a scientifically useful development task. Freeze each cell's
corruption, matched control and runtime before its comparison. Stage one cell at
a time under a separately approved budget; record failures as negative results.
Existing historical evidence can inform design but cannot become fresh evidence.

## Budget scenario, not execution approval

The JSON plan calculates a conservative context-bound scenario: 390 training
rows per task, at most 8192 rendered tokens per row, and the historical $0.40/M
training price for these three models. The reference grid uses four jobs and
18 summed epochs per cell. A later comparison has healthy/damaged/fixed/control
arms; its illustrative three seeds per arm and at most six epochs are a budget
envelope, not a powered seed recommendation. The actual job count and budget
must be frozen after admission and empirical noise measurement.

This envelope is intentionally much larger than a rendered-token estimate.
Before approval, count each model's own template tokens; reject over-context
examples, verify current rates, add inference and contingency, and price any
provider-specific differences. It is separate from the old $5 ledger. Do not
transfer historical estimates or release cancelled holds to fund it.

## B10 coverage

The [local noise result](b10-noise-floor.md) covers BANKING77/Qwen3-0.6B at
500 rows; its 0.0146 pooled exact-match SD does not transfer automatically to
another task, model, backend or size. The completed [210-run study](b9-final-sweep.md)
adds evidence for that local setting and must not be repeated. The rejected
format/content study contributes descriptive strict-score SD 0.0558; it grants
no confirmation seed count.

After admission, use nested development samples at 50, 200 and 390 rows to
separate size from task/model effects. The 390-row healthy/damaged runs can be
reused only when data, recipe, seeds and decoding match the matrix protocol.
The JSON scenario prices the additional 50/200-row cells separately. Three seeds
are a planning placeholder; choose a final count only in the later frozen power
protocol. Report training variability separately from check-example sampling
uncertainty and evaluate the full implemented decision rule. No power simulation
or new confirmation protocol has been run in this package.
