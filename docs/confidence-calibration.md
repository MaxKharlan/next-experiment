# Confidence audit

`bench.calibration` measures whether confidence in the leading candidate matches
how often that candidate is correct. It reads saved predictions and benchmark
truth; it makes no model calls and trains nothing.

```bash
uv run python -m bench.calibration \
  docs/results/dev-routing-20261018/rows.json \
  --split dev --bins 5 \
  --out artifacts/calibration/dev-routing.json
```

The input is a JSON array or JSONL. Each record requires `case`, `variant` (or
`method`), and `ranking`, a list of `[fault_class, probability]` pairs. Every
method must cover the complete selected split exactly once. Unknown cases,
duplicate labels, invalid probabilities and probability mass above one are
rejected. The output file must not already exist.

The selected candidate has the highest probability. Alphabetical label order
breaks ties. This differs from benchmark top-1 when a confirmed cause has been
moved to the front of the saved ranking. Probabilities are not renormalized:
unlisted mass is retained implicitly and is never called a healthy probability.
The candidate is correct when its label occurs in the case's known truth. On a
healthy case no fault candidate is correct, including `other`.

The report contains these metrics for all cases, fault cases and healthy cases:

- Binary Brier score: mean `(confidence - correct)^2` for the selected candidate.
  This is not the multiclass Brier score over the complete fault catalog.
- ECE: the count-weighted absolute difference between mean confidence and
  accuracy within equal-width bins.
- Reliability bins with counts, mean confidence and accuracy. Empty bins have
  null means; the final bin includes probability 1.

Each report records the input and truth hashes and per-case candidate decisions.
These metrics measure candidate confidence, independently of causal confirmation.
An incorrect candidate on a healthy case does not imply a false confirmation.

## First dev audit

On the five cases in the [routing comparison](routing-ablation-dev.md), both
variants have identical rankings and therefore identical metrics:

| Metric | Mixed | Super-only |
|---|---:|---:|
| Candidate accuracy | 3/5 | 3/5 |
| Mean confidence | 0.79054 | 0.79054 |
| Binary Brier score | 0.21186 | 0.21186 |
| ECE, five bins | 0.19054 | 0.19054 |

The three fault candidates are correct; the two healthy cases have incorrect
fault candidates but no confirmed cause. Five examples are too few to estimate
calibration reliably, and ECE depends on the bin choice.

This implements measurement for B12, not a fitted calibration model. Temperature
scaling remains outstanding. Any future temperature must be fitted on dev and
frozen before evaluation on an untouched split. These already inspected held-out
cases cannot support a new claim of untouched evaluation. This tool changes no
agent probabilities, planner decisions or historical benchmark tables.
