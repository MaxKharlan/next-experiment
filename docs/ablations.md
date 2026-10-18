# Cached ablations

This experiment compares five variant families:

- `full`: the EIG planner with Tavily-informed hypotheses.
- `no_tavily`: the EIG planner with hypotheses generated without retrieval.
- `no_causal`: the same EIG policy and Tavily-informed hypotheses as `full`,
  stopping immediately before the first causal intervention.
- `no_planner`: every available probe in registration order, followed by
  causal checks in the original hypothesis order.
- `random_<seed>`: probes and causal checks in seeded random order.

All variants use a USD 1 budget. Variants with causal checks allow at most two
checks, with two seeds per arm, a noise SD of 0.019 and a minimum effect of 0.06.
`no_causal` requests no pairs and incurs no training cost.

The full, no-causal, fixed-order and random-order variants share frozen hypotheses.
The no-Tavily variant has a separate frozen hypothesis generation.
This is one LLM sample per retrieval condition; differences can include
generation variance. Multiple independent preparations are needed to estimate it.

Preparation can incur Nemotron and Tavily charges. Replay uses cached pairs
only and cannot launch training. Missing pairs abort the experiment without
producing a comparison table. Missing full-retrain scores remain unknown.

Probe explanations and verdict narratives are disabled for every variant.
Reported USD costs include the original hypothesis-generation cost and the
estimated training cost of requested pairs, even though replay launches no jobs.
Tavily credits are stored with the frozen hypotheses and excluded from USD totals.
These costs are not directly comparable to the historical benchmark's full
narrative-generation costs.

The causal belief policy multiplies odds for the tested fault class by 3 on
confirmation, by 1/3 on rejection, and by 1 on an inconclusive result.
It preserves probability mass for unlisted causes and does not eliminate an
inconclusive hypothesis. A tested hypothesis is not tested again in that run.

These multipliers are heuristic assumptions, not fitted likelihood ratios.
A higher reported P(true cause) is not evidence of improved calibration.
Beneficial fixes can compensate for other faults, as the historical benchmark's
rank-versus-learning-rate examples demonstrate.

Random seeds produce separate rows. They are repeated orders on the same cases,
not additional independent benchmark cases. Report their variation rather than
multiplying the effective sample size.

The original benchmark tables remain historical results. Reruns of the already
examined held-out split are exploratory follow-up experiments.

## Scheduling after a causal check

Reported beliefs retain tested hypotheses. For choosing the next action,
the EIG planner uses a separate distribution conditioned on hypotheses that
have not been tested. Unlisted probability mass remains in that distribution.
This scheduling distribution is recorded as `planning_probs`; it must not
be interpreted as the reported posterior probability of a cause.

## Replay without causal checks

```bash
uv run python -m bench.ablations --split dev \
  --variants no_causal \
  --out artifacts/ablations/dev-no-causal-v1
```

The output directory must be new. This selection needs only the matching
`with_tavily` frozen hypotheses; it does not read pair results or require
`without_tavily` hypotheses. Use `--variants full no_causal` for a comparison;
the full variant requires its cached pairs. Omitting `--variants` runs every
family. `--variants` applies only to replay, not preparation.

The ablation follows the full method's probe decisions until the first
intervention would occur. A sufficiently high prior can therefore stop it before
any probe. It retains the ranking and probabilities at that point, with status
`inconclusive`, no confirmed cause, no patch and no retrain proposal. Top-1, top-3
and P(true cause) remain measurable. Zero false confirmations and zero successful
fixes follow from this protocol; they are not evidence of better diagnosis.

Protocol version 2 records the no-causal rule and selected variants. Frozen
hypotheses must still match their source fingerprint. Older caches are rejected;
replay does not regenerate them or make API calls. Preparing replacement
hypotheses is a separate, potentially paid operation. Historical results remain
unchanged. The separate routing comparison below enables narratives.


## Super-only routing

`bench.routing_ablation` compares the normal router (`mixed`) with `super_only`.
The latter sends every registered step to Super, including verdicts and steps
that would normally escalate to Ultra. The API and other benchmarks retain their
normal routing unless this option is explicitly selected in `Router`.

Both variants reuse the same frozen, Tavily-informed Super hypotheses and cached
training pairs. Probe explanations and verdict narratives are enabled in both.
The cached ablations above cannot measure routing differences: their hypotheses
already come from Super and their narratives are disabled.

This comparison makes paid Nemotron calls, but starts no training and makes no
Tavily calls. It checks every frozen input and the offline pair trajectory before
creating the provider. Missing pairs or stale hypotheses stop the run. A missing
pair later in a live trajectory also stops it; there is no training fallback.
Use a new output directory:

```bash
uv run python -m bench.routing_ablation \
  --split dev \
  --priors artifacts/ablations/priors-v1 \
  --out artifacts/ablations/dev-routing-v1 \
  --allow-api-calls
```

The priors must match the current source fingerprint and record hypothesis calls
from the configured Super model. Preparing replacement priors through
`bench.ablations --prepare` is a separate, potentially paid operation.

Each case produces a diagnosis and a call log for each variant. `rows.json`
compares rankings, causal status, tokens, LLM costs and cached training use.
`protocol.json` records the models, thresholds, source hashes and input hashes.
Call logs distinguish shared hypothesis costs from new narrative calls, including
billed retries on failure. A failed run writes `INCOMPLETE.txt` and produces no
combined results file. Existing output directories are never overwritten.

Causal status and the patch come from code, not the verdict model. This experiment
primarily measures narrative cost and provides texts for review; it does not
score explanation quality automatically. Different call costs can affect decisions
near the USD 1 budget limit. This budget guides the loop; it is not a hard provider
spending cap. One generation per variant does not estimate generation variance.
These results must not be mixed with narrative-disabled ablation costs. Reruns of
the already inspected held-out cases remain exploratory.

The first dev run is recorded in [Dev routing results](routing-ablation-dev.md), with the raw diagnoses and call logs included.
