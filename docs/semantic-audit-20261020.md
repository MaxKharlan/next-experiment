# Automatic audit of 200 training examples

The user replaced A19's manual review requirement with automatic checking on
20 October 2026. The [acceptance amendment](results/semantic-audit-20261020/acceptance-amendment.json)
records that change. This package compares Nano, Super, three embedding models,
and logistic regression with cleanlab against a constructed reference. It is
not a human annotation study or a calibration of production fault likelihoods.

The frozen selection takes 400 reference queries and 200 disjoint audit queries
from eight BANKING77 intents, using only the pinned upstream training split.
The audit has 100 unchanged labels, 50 substituted labels, and 50 labels wrapped
in JSON in violation of the exact-label contract. Source annotations and the
injection log define the automatic reference. Ambiguity in source intent labels
can therefore affect the results. PolyAI's dataset is
[CC BY 4.0](https://huggingface.co/datasets/PolyAI/banking77).

All methods see the same queries. Nano and Super receive the same three local
Qwen neighbors and the allowed labels, but no source labels or injection flags.
Each request gets one attempt. The prompt, models, decoding, similarity threshold
and cleanlab settings were fixed before their respective measurements. The
combined method is the prespecified union of Nano flags and local Qwen cleanlab
flags. No prompt or threshold was changed after inspecting the failures.

| Method | Wrong-label TP / FP / FN | Precision | Recall |
| --- | --- | --- | --- |
| Nano | 0 / 0 / 50 | undefined | 0% |
| Super | 31 / 0 / 19 | 100% | 62% |
| Local Qwen 0.6B neighbors | 44 / 8 / 6 | 84.6% | 88% |
| NVIDIA Embed 1B neighbors | 48 / 6 / 2 | 88.9% | 96% |
| Provider Qwen 8B neighbors | 48 / 1 / 2 | 98.0% | 96% |
| Local Qwen + cleanlab | 46 / 4 / 4 | 92% | 92% |
| NVIDIA + cleanlab | 46 / 1 / 4 | 97.9% | 92% |
| Provider Qwen + cleanlab | 46 / 1 / 4 | 97.9% | 92% |

The deterministic label-format rule finds all 50 format violations without false
positives. Nano's format flags find none and flag one clean example. Super finds
49 of 50. Two Super requests returned HTTP errors; they remain in the denominator
with no detection credit and with their spending holds retained. All 200 Nano
responses were parseable. Nano often proposes the correct replacement label while
its contradiction flag remains false. For example, `audit-005` is a wrong-cash-
amount query labeled as an unrecognized withdrawal: Nano corrects the label but
says it matches. Super also produces some internally inconsistent flags. These
are measured output failures, not grounds for silently rescoring the primary
endpoint from another field.

Local Qwen and provider Qwen agree on 94.5% of contradiction decisions; NVIDIA
and provider Qwen agree on 96.5%. Different model sizes, representations and
threshold scales prevent treating their vectors or decisions as interchangeable.
All use symmetric normalized document embeddings. The embedding dimensions are
1024, 2048 and 4096 respectively. Cleanlab uses five-fold out-of-fold probabilities
from logistic regression on 400 references plus 150 valid-label audit rows.
No row's own label trains the model producing its probability.

NVIDIA's model is `nvidia/Nemotron-3-Embed-1B-BF16`, revision
`c0c9fea93ea424587517f2c59e20db9f1d6bf615`, licensed under
[OpenMDW-1.1](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16/blob/main/LICENSE).
The local run uses BF16 and SDPA on MPS. Transformers warned that
`apply_yarn_scaling` is an unrecognized rope parameter; this short-query
measurement does not validate long-context behavior. Qwen local uses revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` under Apache 2.0.

Nano's 200 calls cost an estimated $0.00800442 before tax; Super's responses cost
$0.03966210. These estimates use returned usage and the
[provider's published rates](https://nebius.com/services/token-factory/models/nvidia-nemotron-models-inference).
The global ledger charges a further 20% allowance and retains ambiguous holds.
Embedding receipts use the [Qwen embedding tariff](https://nebius.com/services/token-factory/models/qwen-models-inference).
These are not invoices or per-call billing confirmations.

The [combined report](results/semantic-audit-20261020/combined-results/report.json)
contains every method's confusion matrix, failures and Nano/Super disagreements.
[Portable evidence](results/semantic-audit-20261020/evidence/) includes raw provider
responses, request IDs, usage, embedding arrays and runtime manifests. The
[manifest](results/semantic-audit-20261020/manifest.json) binds those files. The
original analysis source is preserved; the current script additionally tolerates
missing fields in the failure-only disagreement path without changing metrics.
The earlier blank human-review form is obsolete under the acceptance amendment.

A19's revised automatic comparison, A30's development outcome comparison and
NVIDIA model review, and A31's baseline comparison now have measured results.
The models remain review aids. This package does not establish downstream
fine-tuning harm, calibrated fault signatures, or independent provider transfer
for the historical recipe classifier.
