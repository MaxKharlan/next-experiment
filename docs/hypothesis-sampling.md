# Repeated hypothesis generation

The live API supports `HYPOTHESIS_SAMPLES=1..5`. The default is one call, preserving
the existing behavior. For example:

```bash
HYPOTHESIS_SAMPLES=3 uv run --group ml --group eval uvicorn apps.api.main:app
```

Starting a live investigation with this setting makes three separate hypothesis
requests with the same case and retrieved evidence. Earlier model answers are
not included in subsequent requests. The existing provider temperature is 0.2;
separate calls can still return identical answers. Tavily retrieval and AnswerCut
preflight are performed once, before hypothesis generation.

The aggregator sums duplicate entries for a fault class within each draw and
averages class probabilities across draws. A missing class contributes zero.
Unlisted probability mass is not redistributed. Classes are sorted by mean
probability, with class name breaking ties, and receive fresh hypothesis IDs.
The highest-prior entry supplies the representative location and rationale;
validated source URLs are combined. Per-draw cleaning retains the existing
mode restrictions and source validation.

For multiple draws, the event report contains a `hypothesis_sampling` step before
`hypotheses`, including all cleaned draws and per-class support counts. These
counts measure agreement, not causal evidence or calibrated confidence. The
planner receives the averaged probabilities and still needs a causal check to
confirm a cause. The report retains the underlying draws so disagreement remains
visible even though the planner groups by fault class rather than location.

Every call and structured-output retry is counted by the existing router log.
Before each additional draw, the sampler checks the remaining investigation
budget. Exhaustion fails the investigation rather than silently using fewer
samples. This check does not predict the next response cost, so it is not a hard
provider spending cap. A provider error also fails the investigation. Existing
failure reports may not include complete LLM billing logs; provider usage remains
the authority for charges on failed runs.

The Python loop also accepts `hypothesis_samples`. Supplying frozen hypotheses
with multiple samples is rejected, preventing hidden requests during cached
benchmarks. Cached ablation commands continue to use their recorded hypotheses.

This implements the repeated-generation option for B14. No paid comparison of
one versus multiple draws has been run, and no accuracy improvement is claimed.
Before enabling it by default, compare both settings on dev with the same causal
pair cache and account for the extra model cost. Consensus can repeat a shared
model error.
