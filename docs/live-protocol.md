# Live Token Factory protocol

The live BANKING77 investigation uses the registered fault6-lr1e-5 case and
config/live-tf-banking77.json.

The configuration defines the check set, noise floor, minimum effect,
training seeds, estimated tokens per example and investigation budget.
These values apply to the existing Qwen3-0.6B experiment and are not universal
defaults for other models or tasks.

Before model calls, the runner checks that train, validation, check and test
exist, are nonempty chat JSONL files, use distinct resolved paths and do not
share identical records. This does not detect semantic duplicates.

Validation is supplied to training. Causal checks are scored on check.
The approved final comparison is scored on test.

Dataset paths, SHA-256 hashes, row counts and protocol parameters are saved
in the investigation report. Files are checked again before causal training
and before an approved retrain.

The retrain proposal retains the original failed job ID and its recipe.
It does not substitute a newly trained control job for the original model.

The planner uses a conservative pair estimate for the current recipe
interventions. Before submission, each arm is priced using its actual recipe,
data fraction and seed count. The pair is not submitted if its estimate exceeds
the remaining budget after recorded LLM costs.

Training costs are estimates from the repository's token-price model, not
provider billing records. Since PR #85, live inference reserves estimated input
and maximum output usage in a durable SQLite ledger before each request. Paid
probes and causal pairs pause for REST/UI approval and resume from saved state.
Unknown outcomes retain reservations. These estimates do not enforce a cap at the
provider; invoice reconciliation, Tavily dollars and an approved full retrain
remain separate. See [the B17 implementation boundaries](next-batch-20261019.md).

Explicit case validation rejects unsupported message roles, blank content and
missing user prompts in all four splits before starting a worker. Nonempty damaged
training labels remain inspectable. Length preflight and answer damage are
separate checks; see [fault 1 semantics](fault-1-semantics.md).

Historical reports and the local-training backend retain their existing
behavior. New live TF investigations require the complete protocol.
