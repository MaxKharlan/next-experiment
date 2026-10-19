# Development check sensitivity — 19 October 2026

The previous fresh answer-cut run remains stopped at `zero_sd`. This follow-up
checks why that development task saturated and tests a separate candidate using
only development data. The historical final test and the prepared fresh final
test are not read. No old training job is repeated.

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
text parser checks all 312 references. Entity IDs are disjoint. The splits share
syntax and slot vocabulary; this is development coverage of a bounded task.
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

## Reproduction

The scripts retain per-example answers, source-family labels, strict scores,
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
