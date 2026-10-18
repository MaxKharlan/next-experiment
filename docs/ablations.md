# Cached ablations

This experiment compares four variants:

- `full`: the EIG planner with Tavily-informed hypotheses.
- `no_tavily`: the EIG planner with hypotheses generated without retrieval.
- `no_planner`: every available probe in registration order, followed by
  causal checks in the original hypothesis order.
- `random_<seed>`: probes and causal checks in seeded random order.

All variants use a USD 1 budget, at most two causal checks, two seeds per arm,
a noise SD of 0.019 and a minimum effect of 0.06.

The full, fixed-order and random-order variants share frozen hypotheses.
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
