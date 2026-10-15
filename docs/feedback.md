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
