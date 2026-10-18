# B20: current dev ablations

The current narrative-disabled agent was replayed on all five dev cases: three
low-learning-rate faults and two healthy jobs. The run contains 70 case/variant
results: full, no Tavily, fixed order, no causal checks, and ten random orders.
The [protocol](results/ablations-dev-20261018/protocol.json),
[score rows](results/ablations-dev-20261018/rows.jsonl),
[all ten random-order tables](results/ablations-dev-20261018/summary.md), and
[provenance](results/ablations-dev-20261018/provenance.json) are saved with the
individual diagnoses, frozen hypotheses and causal-pair/fix caches.

## Main comparison

| Variant | Top-1 on faults | False alarms on healthy | Wrong confirmations | Fix success | Mean attributed USD/case | Cached jobs, total |
|---|---|---|---|---|---|---|
| Full | 3/3 | 0/2 | 0 | 2/3 | 0.122918 | 16 |
| No Tavily | 3/3 | 0/2 | 0 | 2/3 | 0.122045 | 16 |
| Fixed order | 3/3 | 0/2 | 0 | 2/3 | 0.210470 | 24 |
| No causal checks | 3/3 | 0/2 | 0 | 0/3 | 0.002534 | 0 |
| Random order, range over seeds 0–9 | 0/3–3/3 | 0/2 | 0–3 | See below | 0.210470–0.243302 | 24–32 |

Full and fixed order identify the same causes here. Full uses eight fewer cached
jobs and 41.6% less attributed cost. No-Tavily matches full on these five cases;
this single comparison does not establish that retrieval has no value. Retrieval
conditions use separately sampled hypotheses, so generation variability remains
a confounder.

Random seeds 0, 3, 4 and 5 confirm `lora_rank_too_small` on all three low-learning-rate
faults. The remaining six orders identify the correct cause. A larger adapter
can improve an undertrained model without proving insufficient rank was the
original fault. This exposes the causal test's specificity limit rather than
mere sampling noise. Each order reuses the same five cases; the fifty rows are
not fifty independent cases.

Full, no-Tavily and fixed order have measured fix success on two of three fault
cases. The rank fixes selected by the four incorrect random orders lack saved
full-retrain scores: their fix success is unknown, not zero. No-causal cannot
confirm a cause or propose a fix by construction; its zero confirmations and
zero successful fixes are not evidence of superior safety or inferior ranking.
Its fault ranking remains correct on all three cases.

Mean reported P(true cause) is 0.867 for full and 0.691 without causal checks.
The causal update uses heuristic likelihood multipliers; this increase is not
a calibration result.

## Routing ablation: a separate experiment

The [existing dev routing comparison](routing-ablation-dev.md) covers Mixed
versus Super-only with narratives enabled. Both confirmed 3/3 faults with 0/2
healthy false alarms. New narrative costs were USD 0.0161447 and USD 0.0111630
respectively: a 30.9% reduction in that run, or 17.3% including shared hypotheses.
Narrative quality was not scored.

Those routing results were not regenerated in this task and retain their original
source protocol. They complete the report's coverage of the routing question,
but do not belong in the cost ranking above: this replay disables narratives.
The one-versus-three hypothesis experiment is documented separately in
[hypothesis sampling](hypothesis-sampling-dev.md).

## Inputs and spending

The old `priors-v1` cache failed the current source fingerprint check on all ten
dev and sixteen held-out records. Five current with-Tavily records were reused
from the routing experiment; five without-Tavily records were newly generated.
Preparation made five Super calls, 15,558 tokens, with a token-priced cost of
USD 0.0083064. It made no Tavily calls. Replay made no model calls and launched
no training jobs. Prices are calculated from call logs, not a provider invoice.

Table costs attribute the historical hypothesis cost plus estimated training
cost of requested cached pairs. They are not new spending, and summing costs
across variants would count shared inputs repeatedly. Tavily credits are excluded
from USD costs; the frozen with-Tavily inputs retain their retrieval metadata.

All variants use the same USD 1 budget, up to two causal checks, two seeds per
arm, noise SD 0.019 and minimum effect 0.06. Weight probes are disabled. The
production signature table is used; the experimental B16 dev-fit override is
not enabled. The protocol therefore changes only the stated ablation factor.

## Reproduce

From the repository root with the benchmark's local training files and cached
pairs available:

```bash
NEXT_EXPERIMENT_WEIGHTS=0 uv run python -m bench.ablations \
  --split dev \
  --priors docs/results/ablations-dev-20261018/priors \
  --out artifacts/ablations/b20-dev-reproduction
```

The output directory must be new. Input and source hashes are in the provenance
file; fingerprints must still match. A future code change requires a new audit,
not bypassing stale-cache validation. The bundle includes pair/fix caches for
inspection, but does not redistribute the training datasets used to count tokens
and examples. No `--prepare` is needed to replay these frozen hypotheses.

This is a dev analysis on one task, one model and one fault class. Held-out was
not rerun; it has already been inspected historically and is not an untouched
final evaluation. Multiple independent hypothesis preparations and a second
verified fault class remain necessary for broader claims.
