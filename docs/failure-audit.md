# Reproducible benchmark failure inventory

This benchmark inventory makes the saved benchmark's failure cases inspectable without
rerunning models or training. It complements the explanations in
[the benchmark write-up](benchmark.md#error-analysis).

- [Dev inventory](results/failure-audit-20261018/dev/README.md): five cases per method.
- [Heldout inventory](results/failure-audit-20261018/heldout/README.md): eight cases per method.

Each directory includes the normalized source rows and a JSON report with the
source-file hash, truth hash, per-method counts and the complete score row behind
every finding. These are historical scores. They do not measure the current
agent after later belief updates, calibration or repeated hypothesis generation.

The checklist has six wrong confirmations across both splits, as described in
the benchmark write-up. Next Experiment has three measured fix failures: the
cause was correctly confirmed, but the standard one-factor fix did not reach the
reference tolerance. Those failures must not be described as incorrect diagnoses.

## Generate an inventory

```bash
uv run python -m bench.failure_report \
  artifacts/bench/dev/rows.jsonl --split dev \
  --out artifacts/failure-audit/dev

uv run python -m bench.failure_report \
  artifacts/bench/heldout/rows.jsonl --split heldout \
  --out artifacts/failure-audit/heldout
```

Inputs are JSON arrays or JSONL files in the `bench.score.Row` format. Each method
must cover the full selected split once; unknown cases, mismatched truth,
duplicate rows and inconsistent top-1 or false-alarm flags are rejected. Existing
output directories are never overwritten. No provider APIs are imported or used.

## Categories

- `wrong_top_candidate`: the first ranked cause is wrong on a fault case.
- `cause_not_confirmed`: the method ends inconclusive on a fault case.
- `false_alarm_on_healthy`: a cause is confirmed on a healthy case.
- `wrong_primary_confirmation`: the selected confirmed cause is outside the truth.
- `wrong_confirmations`: the scorer recorded one or more incorrect confirmations,
  including secondary ones. The report retains both affected-case counts and the
  total number of wrong confirmations.
- `measured_fix_failed`: a confirmed diagnosis has a recorded fix metric and
  reference, and the saved scorer marks the fix unsuccessful.

Categories overlap. An inconclusive diagnosis is not automatically a measured
fix failure. Missing fix metrics are listed separately for confirmed cases.
The inventory does not infer which secondary cause was wrong when the score row
contains only a count, or infer the mechanism behind a failure. Use the original
investigation reports for that analysis.

The historical benchmark contains only low-learning-rate fault cases. This
inventory does not close evaluation of answer truncation or other fault classes,
and it does not replace B9 fault coverage or a new heldout experiment.

The external-source collection required by B21 is maintained separately in
[Public fine-tuning reports](public-finetuning-cases.md).
