# Probe budget accounting

Before executing a selected probe, the investigation loop compares its cost
estimate with the remaining budget after recorded LLM calls. An unaffordable
probe produces a saved stop step with its name, estimate and remaining budget.
The rule applies to EIG, fixed and random probe policies.

After execution, the loop deducts `ProbeResult.cost_usd` and records the estimate,
reported cost and remaining budget in the probe step. Later causal checks use
that reduced budget. If a probe exhausts or exceeds the budget, the loop skips
its LLM explanation. Existing zero-cost probes continue to cost zero.

A probe result must report its own non-LLM costs; calls already recorded in the
router's LLM log must not be included again. Estimates and reported costs must
be finite and nonnegative. A probe can exceed its estimate while executing;
this guard cannot undo that spending or enforce a provider-side hard cap.
LLM explanations still have no advance token-cost reservation, so a positive
remaining balance does not guarantee that the next explanation fits within it.

Tests use synthetic paid probes and fake model responses. They cover all three
policies, actual-cost deduction, blocked causal checks, estimate validation and
skipping explanations after an overrun. No paid requests are required.
