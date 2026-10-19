# Format/content development experiment, 19 October 2026

This experiment follows the stopped history-task pilot and the negative transfer
screen of six old adapters. It trains twelve new adapters. Historical protocols,
results and final-test data remain unchanged.

## Frozen design

Four arms, each with training-order seeds 1301, 1302 and 1303:

| Arm | 72 common training anchors | 144 training histories |
| --- | --- | --- |
| format | Direct serialization of supplied fields | Converted to direct serialization |
| healthy | Clean | Clean |
| cut | Clean | Half the answers cut at half their characters |
| wrong | Clean | The same half assigned cyclically wrong product and priority, valid JSON |

Each validation set contains 12 anchors and 24 corresponding examples. The check
has 24 unseen direct serializations and 48 unseen histories. Splits use distinct
entities and normalized histories, a new random seed and new IDs/dates, but share
a synthetic grammar. Clean history labels are verified by a text-only replay
parser; direct labels are checked against the supplied field values. This is an
automated single-author review, not independent human annotation.

Qwen3-0.6B LoRA uses two epochs, learning rate 1e-4, batch 8, rank/alpha 16,
dropout .05 and an 8192-token context without packing. All jobs start from the
same base; the format arm is a diagnostic learning control and is not compute
matched because its inputs are shorter.

## Separate measurements

1. Format check: raw schema validity and strict full-object exact match.
2. History generation: strict exact match with greedy decoding, thinking off and
   a fixed 192-token limit. A preplanned diagnostic removes only a single
   canonical Markdown JSON fence. Invalid/unparseable answers remain wrong;
   every example stays in the denominator.
3. Bounded content discrimination: conditional log probability of four fixed,
   equal-token-length canonical JSON completions. Only product and priority
   differ; other fields are supplied in all candidates. The correct completion
   must be the unique maximum (ties within 1e-6 count wrong). This measures
   two-field discrimination, not full open-ended reasoning.

The base model is measured once without a strict-JSON admission gate. All twelve
new trainings are planned before base measurement, so base formatting cannot
cancel the requested training test.

Before any model measurement, the protocol fixes these gates: mean direct
serialization exact >= .90 for format and healthy; healthy choice accuracy in
[.35, .95]; healthy-minus-format and healthy-minus-wrong choice gaps >= .10;
healthy-minus-cut strict-history gap >= .08. Each arm must have three distinct
weight digests. Only after every gate passes does pooled healthy/cut seed noise
enter the inherited power rule (nonzero SD, candidate counts 3–8, 1,000 Monte
Carlo trials, effect .16, minimum effect .08, lower 95% MC bound >= .80).
A pass permits preparation of a new confirmatory protocol; it does not authorize
selection on a final test. A failed gate ends this design without adaptive seeds.

## Accounting and execution

The new cohort copies the complete six-operation spending ledger from
`ticket-updates-development-20261019-v1`, with $0.2710464 previously estimated
before tax. The new ledger is the sole successor, under the inherited $5 cap.
Training reservations use the longest rendered example times rows times two
epochs, at $0.40/M tokens. The authenticated Token Factory price table was
rechecked on 19 October: Qwen3-0.6B LoRa 8K, eu-north1, effective 1 December 2025.
Another $1 is held out of the training admission calculation for future
inference and $.50 for contingency. Provider token counts determine settled
estimates; these are not invoices.

A cohort process lock prevents concurrent commands. Every paid attempt is
recorded before upload/submission. Unknown outcomes retain their ledger hold and
must be reconciled before any retry. Source, runtime, datasets, candidate choices
and injection records are hash-bound before evaluation. Existing complete scores
are checked rather than regenerated on resume. No service/UI classifier changes
are part of this experiment.

## Execution result

Protocol SHA-256:
`28ef76e4521cef83040423722ef6265624b019de194af098104a9632ab2d8f09`.
Source commit: `330841d`. The complete series reserves $1.1311488 before tax.
All twelve provider jobs were accepted once. Evaluation is pending completion.

| Arm | Rendered training tokens per epoch | Maximum example tokens | Three-job reservation |
| --- | ---: | ---: | ---: |
| format | 61,932 | 292 | $0.1513728 |
| healthy | 97,818 | 630 | $0.3265920 |
| cut | 96,184 | 630 | $0.3265920 |
| wrong | 97,818 | 630 | $0.3265920 |

All check histories are balanced across the three families (16 each). Candidate
position is shuffled and frozen; candidates are scored independently, not shown
as an ordered multiple-choice list to the model.

Reproduce preparation in a new artifact directory with the frozen source commit
and runtime. The generator fixes the RNG; path fields in a regenerated protocol
will differ. Never submit against the historical ledger again.

```bash
uv run python -m scripts.format_content_development prepare \
  --root artifacts/format-content-development-20261019-v1 \
  --old artifacts/ticket-updates-development-20261019-v1
uv run python -m scripts.format_content_development submit \
  --root artifacts/format-content-development-20261019-v1 --env-file /path/to/.env
uv run python -m scripts.format_content_development base \
  --root artifacts/format-content-development-20261019-v1
uv run python -m scripts.format_content_development collect \
  --root artifacts/format-content-development-20261019-v1 --env-file /path/to/.env
uv run python -m scripts.format_content_development score \
  --root artifacts/format-content-development-20261019-v1
uv run python -m scripts.format_content_development analyze \
  --root artifacts/format-content-development-20261019-v1
```

Do not run the paid submission command merely to reproduce analysis. Published
predictions, candidate scores and data are sufficient for rescoring without a
provider call. The terminal decision prevents further scoring/submission in the
same cohort.

To verify a published bundle in a different checkout, without local model files
or credentials:

```bash
uv run python -m scripts.report_format_content --verify \
  --out docs/results/format-content-development-20261019
```

This rechecks data and training-file hashes, all 936 generated-answer metrics,
624 four-choice decisions and the stopping decision. It does not rerun model
inference or independently recover weight bytes from a weight hash.
