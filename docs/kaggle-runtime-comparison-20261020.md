# Same checkpoint on MacBook, AI Cloud and Kaggle

The frozen Qwen3-0.6B base checkpoint produced identical saved evaluation records
on all 64 development queries across MacBook MPS, AI Cloud L40S and Kaggle T4.
Each environment scored 9/64 exact answers (14.0625%); every answer ended on EOS,
and none reached the generation limit. Answers, scores, prompt and answer token
counts, example identities and stopping flags all agree.

| Environment | Exact answers | Different records versus Kaggle |
| --- | ---: | ---: |
| MacBook MPS | 9/64 | 0/64 |
| AI Cloud L40S | 9/64 | 0/64 |
| Kaggle T4 | 9/64 | Reference |

The [verification record](results/kaggle-evaluation-v3-20261020/comparison.json)
contains input, evaluator and checkpoint hashes, source receipts, recomputed
metrics and the empty disagreement lists. The checkpoint revision is
`c1899de289a04d12100db370d81485cdf75e47ca`. Generation was greedy, with thinking
disabled and a limit of 32 new tokens. The CUDA evaluations used float32 with
TF32 disabled and math SDPA. The Kaggle worker checked every checkpoint file
digest before loading the model; the local cached checkpoint independently
matches that manifest. Remote model bytes were not exported.

Kaggle ran Python 3.13.15, PyTorch 2.14.1 and CUDA 13.0 on a T4.
All [46 preflight checks](results/kaggle-evaluation-v3-20261020/measurements/preflight.json)
passed: bundle hashes, pinned package versions, pip compatibility, bundled and
lazy loader imports, GPU discovery, float32 matrix multiplication and math SDPA.
The application's documented Python support remains 3.11–3.12; this run checks
only the standalone evaluation notebook on 3.13.

The first Kaggle attempt failed in `ensurepip`. The second installed its runtime
but lacked `pydantic`, which the structured metric module imports. Both failed
before model evaluation. V3 installed the complete evaluator dependencies in an
isolated environment and removed inherited Python paths. The frozen worker,
check set and evaluator files are byte-identical to V2. Historical failures and
their [pre-generation diagnosis](results/kaggle-evaluation-v2-20261020/failure.json)
remain recorded.

The [downloaded measurements](results/kaggle-evaluation-v3-20261020/measurements)
include all three prediction sets. The local client uploaded nine evidence
files to the existing Object Storage bucket and verified their bytes by reading
them back; see the [export receipt](results/kaggle-evaluation-v3-20261020/storage-export.json).
The Kaggle notebook contains no account or storage credentials.

This completes A20's same-checkpoint comparison for the frozen base model and
64-query development set. It does not establish equality of logits, training
results, other checkpoints or other decoding settings. No new training or
final-test evaluation was performed.
