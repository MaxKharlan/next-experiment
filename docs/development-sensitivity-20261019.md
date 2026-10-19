# Development check sensitivity — 19 October 2026

The previous fresh answer-cut run remains stopped at `zero_sd`. This follow-up
checks why that development task saturated and tests a separate candidate using
only development data. The historical final test and the prepared fresh final
test are not read. No old training job is repeated.

The fixed-adapter screen completed and **rejected** the candidate with
`insufficient_damage_sensitivity`: healthy mean 26.85%, damaged mean 39.81%.
The candidate's separate paid pilot had already stopped at its strict base gate.
Additional provider spend is **$0**. No pair seeds were selected and no final
test was opened. [Machine-readable summary](results/development-sensitivity-20261019/summary.json).

## Historical ceiling audit

All 612 saved adapter predictions were rescored and agree with their recorded
102/102 results. Each of 102 references was then changed one field at a time;
all 510 wrong-field answers and all 102 truncated answers were rejected. All
102 unchanged references passed. The strict evaluator detects those errors.

The historical check contains 18 product/priority/callback combinations, all
present in training. Normalizing ticket IDs and dates leaves 42 user prompts in
each split; the outer wording differs but the extraction pattern is shared.
The 96 templated rows and six paraphrases do not provide broad language coverage.

A separately frozen base-model evaluation scored **0/102 strict exact match**.
All 102 answers wrapped JSON in a Markdown code block. A post-hoc diagnostic
removed only that wrapper and found all 102 contents correct. This diagnostic
is not a replacement metric and is never used to change the historical stopping
rule. It shows that once training teaches the bare-JSON convention, this check
has little content difficulty left. It does not establish that corruption has
no effect on other data.

## Separate candidate and admission rules

`ticket-updates-development-20261019-v1` contains 192 train, 48 validation and
72 check histories. Each names one target ticket, with a second ticket as a
distractor. The three equally represented families contain corrections,
interleaved updates to the other ticket, or conflicting void updates. A separate
text parser checks all 312 references. Entity IDs are disjoint. A separate input
audit found no cross-split overlap after replacing entity IDs by their role in
the history; all 312 normalized histories are unique. The splits share
syntax and slot vocabulary; this is development coverage of a bounded task.
These check histories are now development data permanently and cannot become a
future untouched final test.
One session authored both generator and parser; there was no independent human
review. No final test is created.

Before candidate scoring, protocol
`614f609360032f9481d8e765cacb8d5f4e5a7ddcce456b5d2c9db39c48cda590`
fixed one base evaluation and at most six new trainings. The base must score
between 0.10 and 0.90 under strict JSON exact match before any payment. Later
admission requires healthy mean in [0.10, 0.95], at least 0.05 gain over base,
at least 0.08 healthy-minus-damaged gap, distinct weights, positive pooled SD
and the prespecified power gate. No adaptive repeats or imported variance floor
are allowed. Passing would require a separate confirmatory protocol.

The six-job conservative reserve is $0.290304 using the previously verified
same-day $0.40/million token price. The full SQLite ledger was copied through
SQLite backup: six settled historical operations, $0.2710464 used, cap $5.
The successor ledger is in `artifacts/ticket-updates-development-20261019-v1/`;
new work must never use the old ledger in parallel. $1 for future inference and
$0.50 contingency remain outside the pilot training estimate. Costs are token
estimates before taxes, not reconciled invoices.

The base evaluation scored **0/72** strict exact match, so this candidate's paid
phase stopped with `base_floor_or_ceiling`. No jobs were submitted. A post-hoc
format diagnostic found 71 fenced answers and 29/72 correct contents after
unwrapping, compared with 102/102 on the historical check. This supports increased
content difficulty but does not override the frozen strict-score gate.

## Fixed-adapter sensitivity screen

The historical base audit exposed a format confound in the candidate's strict
base-score admission criterion. The candidate gate is retained. A separate,
free exploratory screen asks whether the already-trained adapters, which learned
bare JSON, expose content failures on the candidate histories.

Protocol `ecdcc22c285db28098a957c013fb22f279f31c22fb0c9000b59c15c3daff7a11`
fixes all six existing adapters and the 72 candidate check rows before any
screen scores. Healthy mean must be in [0.10, 0.95], the mean damage gap at least
0.08 and pooled SD positive. The screen has no training action and selects no
seeds. It cannot reopen either predecessor protocol or authorize a paid API
trace. Its scores are exploratory development measurements of fixed historical
weights, not fresh model comparisons or final-test evidence.

## Screen result and limits

| Historical training arm | Seed 1101 | Seed 1102 | Seed 1103 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Healthy | 26/72 (36.11%) | 19/72 (26.39%) | 13/72 (18.06%) | 26.85% |
| Damaged | 29/72 (40.28%) | 33/72 (45.83%) | 24/72 (33.33%) | 39.81% |

All six evaluations exited successfully and preserved 432 predictions. The
strict scores were recomputed from the saved answers. Healthy-minus-damaged is
-0.12963, below the required +0.08. Pooled seed-level SD is 0.07774 on these
histories. The screen therefore rejects this candidate for a confirmatory run;
no power simulation, seed selection or paid follow-up occurred.

The variance describes existing weights on this new check. It cannot be imported
as the noise of models trained on ticket histories. These adapters were trained
on the old simple extraction task; no model was trained on the new histories.
The reverse gap is exploratory transfer evidence, not a demonstration that
corruption improves generalization. One healthy seed also failed the output
schema on 20/72 rows and hit the 192-token limit on five; none of those examples
were removed or rescored under a different primary metric.

This work removes the historical ceiling but does not establish a development
design with a useful quality loss caused by corruption. A later design must
separate output-format learning from content learning and measure degradation
with newly trained development repeats before a reference/control API run.
The old protocol, this stopped paid pilot and this rejected free screen remain
separate terminal records. The prepared historical fresh final test remains
protected, and all candidate check histories remain development-only.

The [35 specificity controls](answer-cut-specificity-controls-20261019.md) also
supply clean-prefix and competing-malformation counterexamples for future
calibration. Live signatures, priors and classifier deployment were not changed.

## Reproduction

[Saved evidence](results/development-sensitivity-20261019/) includes separate
`audit`, `pilot` and `screen` records. The scripts retain per-example answers,
source-family labels, strict scores,
dataset and adapter hashes, runtime pins and protocol digests. Use a new output
directory when reproducing preparation. The candidate dataset generator needs
no model or provider access:

```python
from pathlib import Path
from bench.tasks.ticket_updates import build

build(Path("artifacts/ticket-updates-reproduction"))
```

Scoring requires the pinned local snapshot and an exclusive MPS slot coordinated
with B9. The original MPS handoff was reconfirmed in its chat before this work.
Only one model-evaluation process runs at a time. T=1 and the offline status of
the experimental classifier are retained.

## Validation and resources

Ruff and formatting passed. Full local pytest: **826 passed, 8 skipped**.
The 38 control artifacts reproduced byte-for-byte. Both base evaluations and
all six adapter evaluations exited with code 0. The MPS slot was released and
its availability reported to the B9 owner after completion.

The inherited six ledger entries remain settled at $0.2710464 estimated total;
$4.7289536 remains under the $5 cap. No ambiguous provider operations or new
reservations exist. B9's completed sweep was not repeated; no automation was
created. Fresh reference/control comparisons, persisted paid approvals/restart
recovery, fresh final-test evaluation and the 3×3 task/model matrix remain open.
