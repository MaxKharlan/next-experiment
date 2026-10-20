# Feedback log

Running notes on Nebius Token Factory, Nebius AI Cloud, NVIDIA Nemotron and Tavily for the Devpost feedback. One entry per finding: date, product, what happened, what we expected.

## 2026-10-14 · Nebius AI Cloud · PermissionDenied before the first payment

Creating a bucket and an access key returned `PermissionDenied` in both the console and the CLI. The actual cause was the missing $25 first payment, but the error did not mention billing. After the payment the same commands worked.

## 2026-10-14 · Nebius AI Cloud · two kinds of service account keys

Access keys from `nebius iam v2 access-key` work only for Object Storage. The Nebius CLI and Serverless Jobs need a separate authorized key (`nebius iam auth-public-key generate`). The console does not make this difference obvious, and we first tried the S3 keys with the CLI.

## 2026-10-15 · Token Factory · structured output docs

The "Structured output & JSON" page passes the JSON Schema directly as `json_schema`. The API rejects that with a 422 and expects the OpenAI shape `{"name": ..., "schema": ...}`. With that shape, Nemotron 3 Nano, Super and Ultra all returned schema-valid JSON on the first attempt.

## 2026-10-15 · Token Factory · Nemotron 3 replies

All three Nemotron 3 models work through the OpenAI-compatible API. Plain-text replies can start with blank lines, and a one-word answer still used dozens of tokens, which looks like hidden reasoning counted in the usage. The structured smoke test cost $0.00017 on Nano, $0.00063 on Super and $0.00427 on Ultra per call.

## 2026-10-15 · Token Factory · a job waiting for capacity looks like a stalled job

Three LoRA jobs (two on Qwen/Qwen3-0.6B, one on meta-llama/Llama-3.1-8B-Instruct; 200 chat examples, about 11k tokens) reached the "Training started" event and then stayed `running` with 0 steps and no checkpoints for 35 minutes to 2 hours. There was no error. Support explained that the jobs were waiting for capacity because the cluster was busy. Nothing in the API or the console shows this: the event says training started, `trained_steps`, `total_steps` and `estimated_finish` stay empty, and the console shows "Progress 0%, ETA: Calculating". Once a job gets a GPU these fields fill in. We expected a `queued` status or a "waiting for capacity" event, ideally with a queue position. Support cancelled the jobs and confirmed there was no training charge.

## 2026-10-15 · Token Factory · cancellation waits for a scheduling slot

`POST /v1/fine_tuning/jobs/{id}/cancel` returns the job with status `running`, and nothing changes until the job gets a scheduling slot. For two waiting jobs the "Finetuning was cancelled" event came 33.5 minutes after the request. The console has no "cancelling" state either. We expected a cancel to apply right away to a job that is still waiting, or at least a visible `cancelling` status.

## 2026-10-15 · Token Factory · model lists disagree

The post-training docs and the price list include small models such as Qwen/Qwen3-0.6B, Qwen3-1.7B and Llama-3.2-1B, but the console model picker offers only larger ones, starting from Meta-Llama-3.1-8B-Instruct. The API still accepts jobs for the missing models. The docs give the ID meta-llama/Meta-Llama-3.1-8B-Instruct; the API returns 422 "Model doesn't exist" for it and accepts meta-llama/Llama-3.1-8B-Instruct, which is what the copy button in the picker gives.

## 2026-10-15 · Token Factory · fine-tuning prices are hard to find

Fine-tuning prices per 1M tokens are only on the Billing > Prices page after login, and that page loads in parts behind "Show more prices". The public model catalog, its JSON (`/api/public/models_info`) and the API do not have them. The job overview's Prices block showed an empty "Fine-tuning per 1M tokens" field for a Qwen3-0.6B job. Billing > Usage lagged more than an hour and a half behind finished jobs.

## 2026-10-15 · Token Factory · what a finished LoRA job returns

A job that got a GPU right away finished in 7 to 17 minutes and was easy to inspect. `result_files` hold an adapter in the standard PEFT format (`adapter_model.safetensors`, `adapter_config.json`) with the tokenizer and chat template, and the adapter config shows the exact LoRA setup. The job overview shows seed 0 as "—".


## 2026-10-16 · Token Factory · LoRA checkpoints do not load with PeftModel.from_pretrained

`adapter_model.safetensors` from a Token Factory checkpoint names its weights `model.layers.0.mlp.down_proj.lora_A.weight`, without the `base_model.model.` prefix that PEFT writes and expects. `PeftModel.from_pretrained` only warns about missing adapter keys and loads none of them. Because `adapter_config.json` sets `init_lora_weights: false`, the LoRA layers then hold random weights: a Qwen3-0.6B checkpoint answered with strings like "结结结结". After adding the prefix and loading the weights with `set_peft_model_state_dict`, the same checkpoint scored 0.10 exact match. We expected the files to load with the standard PEFT call, or the docs to show how to load them.


## 2026-10-16 · Token Factory · the job seed has no effect

Two Qwen3-0.6B LoRA jobs with the same data and hyperparameters and `seed` 0 and 1 (ftjob-5aa5f03e…, ftjob-abe4e163…) produced identical checkpoints: train_loss 1.1466, 0.3383, 0.1066 and valid_loss 0.677, 0.5472, 1.0787 in both, and the same exact match. The same held for the pair at learning rate 1e-5. The API stores and returns the seed, so it looks applied. We expected different data order or initialization per seed; to get independent repeats we now shuffle the training file ourselves.

## 2026-10-16 · Token Factory · training is reproducible to the bit

A Qwen3-0.6B LoRA job submitted in the evening (ftjob-ee9302e1…) used the same training file bytes and hyperparameters as a job from the morning (ftjob-5aa5f03e…). Its `adapter_model.safetensors` has the same SHA-256, so its exact match on a 200-query held-out set is the same too, 0.425. This is useful: a saved run can be reproduced exactly, and a benchmark rerun gives the same numbers. Together with the ignored `seed`, it also means repeats of a job differ only if their data does. Token Factory does keep the order of the lines in the training file. The same recipe trained on the file shuffled with seed 0 and with seed 1 gave different weights: exact match 0.06 and 0.04, and 0.24 and 0.30, on 50 validation queries. So shuffling the file ourselves gives independent repeats. We expected a working `seed` for that.

## 2026-10-20 · AI Cloud · isolate package constraints in NGC jobs

Job `aijob-e00f6pqn9j8qsps20h` used the NVIDIA PyTorch 25.05 image and an isolated Python environment. The installer still inherited the image's pip constraint file, which required NumPy 1.26.4 and conflicted with our frozen NumPy 2.4.6. The job stopped before model download or training. Clearing `PIP_CONSTRAINT`, `PIP_BUILD_CONSTRAINT` and `PIP_REQUIREMENT`, and using `PIP_CONFIG_FILE=/dev/null` for that environment's installation, allowed the same pinned dependencies to install. A Serverless Jobs example showing how image-level pip configuration interacts with a virtual environment would help. This is an integration constraint, not evidence that the package versions are interchangeable. [Failure record](results/pretrained-code-cuda-fp32-20261020/decision.json).

## 2026-10-20 · AI Cloud · export bytes to the mounted bucket

Job `aijob-e00s96et8m5pkns9m0` completed 64 base-model predictions, then stopped during `shutil.copytree` into the Object Storage mount. File contents were present, but the metadata-copy step failed. Downloading the two JSON files confirmed the saved predictions and metrics. Our corrected exporter copies bytes without chmod, timestamps or other filesystem metadata; it rejects symlinks before copying. A mount example that distinguishes object contents from POSIX metadata would prevent this failure. The next job reached adapter training with that exporter. [Collection record](results/pretrained-code-cuda-cleanpip-20261020/collection.json).

## 2026-10-20 · Token Factory · training access and inference access differ

The existing Llama-3.2-1B-Instruct job `ftjob-820c72abb1814c2da596103bc39c2fa8` succeeded and provides an adapter and tokenizer. Its `fine_tuned_model` is null, and the authenticated inference model list does not contain this 1B model. The export does not contain base weights. This leaves local generation dependent on separate access to the base model, even though training was accepted. An explicit, model-specific evaluation route in the completed job view would help. We used its exported tokenizer for token budgeting; that does not establish model quality. [Access record](results/matrix-tokenizer-costs-20261020/llama-provider-access.json).
