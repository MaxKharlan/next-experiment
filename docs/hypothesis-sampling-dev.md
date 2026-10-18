# One versus three hypothesis calls on dev

Five dev cases were evaluated with the same cached causal pairs and saved Tavily
context. Each case received three new Super calls. The first draw also served as
the one-call variant, making this a paired comparison with a shared first sample.
Three-call priors were aggregated by the implemented class-probability mean.

| Metric, five cases | One call | Three calls |
|---|---:|---:|
| Fault top-1 | 3/3 | 3/3 |
| Correct causal confirmations | 3/3 | 3/3 |
| False confirmations on healthy cases | 0/2 | 0/2 |
| Cached training jobs attributed | 16 | 16 |
| Hypothesis generation cost | $0.0120633 | $0.0362916 |
| Estimated cached training cost | $0.60192 | $0.60192 |
| Mean P(true cause), fault cases | 0.82634 | 0.76402 |
| Top-candidate binary Brier, all cases | 0.14540 | 0.11633 |
| Top-candidate ECE, five bins | 0.18947 | 0.31648 |

Three calls cost 3.01 times as much for hypothesis generation. They did not change
final diagnosis statuses or the number of cached jobs. Brier improved while ECE
worsened; five cases cannot support a reliable calibration conclusion. Mean
P(true cause) alone is not a calibration metric.

Actual new spending was approximately $0.0362916 across 15 model calls, computed
from token logs and configured prices. Do not add the two variant costs: the
one-call variant shares its calls with the three-call variant. No new training
or Tavily requests were made. Pair cache hashes were checked before and after.
The attributed training costs describe benchmark resource use, not new charges.

Both variants used a USD 1 investigation budget, at most two causal checks, two
seeds per arm, noise SD 0.019 and minimum effect 0.06. B25, probe explanations and
verdict narratives were disabled in both. Full-retrain effectiveness was not
evaluated. All draws, their call logs, aggregated hypotheses and per-case results
are saved under `docs/results/dev-hypothesis-sampling-20261018/`; `protocol.json` records source and pair-cache hashes.

This is a single paired replicate on dev, with three fault and two healthy cases.
It neither estimates repeated-run variance nor validates the answer-truncation
fault. Keep one call as the default; this run shows no diagnosis improvement from
three calls. A larger, repeated comparison would be needed to justify changing it.

[Protocol](results/dev-hypothesis-sampling-20261018/protocol.json) · [Per-case predictions](results/dev-hypothesis-sampling-20261018/predictions.json) · [Confidence audit](results/dev-hypothesis-sampling-20261018/calibration.json).
