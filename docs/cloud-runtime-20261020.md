# Serverless execution and numerical qualification

A pinned Qwen3-0.6B evaluation completed on an L40S Serverless Job and returned
JSON through Nebius Object Storage. The connector checks job name, project,
terminal state and output mount before retrieving a completed result. It reserves
the full configured duration before submission and retains uncertain outcomes.
Repeated submission of the same saved request returns its existing job ID.

Three attempts are preserved:

| Attempt | Outcome |
| --- | --- |
| `aijob-e00q4ycj4fw0mwy2zp` | Failed before model measurement: the NGC image's torchvision was incompatible with the separately pinned torch |
| `aijob-e00mgrf16jhgmtae6s` | Isolated venv completed; identical tokens, but maximum CUDA/MPS region-mean NLL difference 0.003364874 exceeded the 0.0001 gate |
| `aijob-e00efctwj39ghty18g` | Explicit IEEE float32 and math SDPA passed the unchanged gate against MPS and CPU |

The third attempt disables TF32 and the flash/memory-efficient SDPA backends.
Maximum NLL differences were 0.0000207663 versus MPS and 0.0000194312 versus CPU;
maximum EOS probability difference was 0.0000002384 for both. All twelve token
hashes and region token counts agree. The experiment changes both precision and
attention settings, so it does not isolate which setting caused the original
mismatch. [Raw result and decision](results/cloud-precision-20261020/decision.json)
retain the configuration and fixed gates.

This qualifies those teacher-forced computations. It does not establish training
backend equivalence or equality of generated answers. Separate frozen notebooks
and CUDA cohorts compare generation. The original unsubmitted default-CUDA
matrices remain blocked; the new FP32 matrices carry explicit amendments and
new manifests with unchanged data, seeds and recipes.

The implementation uses the existing authenticated CLI session and the job's
managed runtime identity. Dedicated service-account-key onboarding is not
complete. S3 credentials remain local; native bucket IDs mount output prefixes
inside jobs. The two completed qualification VMs were confirmed absent after
termination. Job records and output artifacts remain available.

Each qualification retains its $4 reservation pending provider billing. The
separate cloud cap is $30; estimates are not invoice settlement. Prices and
reservation assumptions are stored in each protocol.

PyTorch documents [TF32 precision controls](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
and the [global TF32 environment override](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html).

The first qualified code-matrix job, `aijob-e00f6pqn9j8qsps20h`, failed before
model download or training: the image exported a pip constraint for NumPy 1.26.4,
while the frozen project requires 2.4.6. The corrected bootstrap removes inherited
constraint/requirement variables and disables pip config only for installation
inside the isolated venv. The new code and data manifests verify unchanged worker,
data, package versions, seeds and recipe. The failed attempt retains its $6 hold.
