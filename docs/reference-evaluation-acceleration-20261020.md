# Qwen reference evaluation on Cloud

The owner requested faster reference evaluation and supplied an updated Cloud
balance. Before the first local prediction, the waiting MPS scorer was stopped.
The original training protocol and its bound source files remain available.

Each Qwen model gets one L40S job after its twelve training jobs finish and its
adapters are downloaded. The two evaluations can run concurrently. Each job has
a two-hour timeout and a $6 reservation; the acceleration allocation is capped
at $12. Earlier Cloud holds remain in their original ledger pending billing.
An owner-reported balance does not replace provider billing evidence.

The CUDA evaluation uses the same model revisions, adapter bytes, data, package
versions, float32 weights, greedy decoding and token limits. It disables TF32
and fused SDPA. A separate protocol records the device change before generation.
The earlier 64-example base-model comparison does not establish equivalence for
all of these adapters.

For each model and task, the evaluator records base validation and check scores,
scores all four adapters on validation, selects the highest score with grid index
as the tie-breaker, then evaluates only that adapter on check. This produces
2,506 records per model and 5,012 in total. These remain development results.

The dependency preflight checks imports, versions, CUDA and all packaged file
hashes before generation. The collector verifies prediction identities, metrics,
selection, counts and the output manifest. Failed and partial results are retained;
generation is never retried automatically.

Execution:

- `scripts.accelerate_reference_evaluation` packages and submits the jobs once.
- `scripts.collect_reference_evaluation` collects terminal results and verifies them.
- Local receipts and the separate allocation ledger live under
  `artifacts/remaining-queue-20261020/qwen-acceleration`.

The one- to two-hour estimate is provisional until the first measured throughput
is available; provisioning and dependency installation add time.
