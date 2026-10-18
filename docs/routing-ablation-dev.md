# Dev routing comparison, 18 October 2026

Five cases: three low-learning-rate faults and two healthy jobs. Both variants
confirmed all three faults and confirmed no cause on either healthy case.
Rankings and diagnosis statuses matched on all five cases.

| Cost for five cases | Mixed | Super-only |
|---|---:|---:|
| New narrative calls | $0.0161447 | $0.0111630 |
| Including shared hypothesis generation | $0.0288155 | $0.0238338 |

New narrative calls cost 30.9% less with Super-only in this run. Including the
shared hypothesis cost, the reduction is 17.3%. Each variant made 16 new calls.
The mixed variant called Ultra for three verdicts; all other calls used Super.

The old five hypothesis files were stale. New Super hypotheses were generated
using the saved Tavily evidence, with retrieval source hashes retained in the new
files. Preparation cost $0.0126708. Total new LLM cost for preparation and both
variants was $0.0399785, across 37 calls. Costs are computed from token logs and
configured prices, not a provider invoice. No new training or Tavily calls ran.
Each variant used results representing 16 cached training jobs, estimated at
$0.60192; that is attributed benchmark cost, not new spending.

Weight probes were disabled in both variants. Narrative quality was not scored.
This is one sample per variant on dev only; it does not establish a general
quality advantage or cost reduction. The experiment reuses frozen hypotheses
and causal metrics, and causal status is determined by code.

The [saved protocol](results/dev-routing-20261018/protocol.json), [per-case rows](results/dev-routing-20261018/rows.json) and individual diagnosis and call files are included under `docs/results/dev-routing-20261018/`.
