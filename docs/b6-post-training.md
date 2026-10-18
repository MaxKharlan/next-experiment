# B6: price and duration of Token Factory fine-tuning jobs

Measured on 2026-10-15 with `scripts/measure_post_training.py`. The Llama-8B fallback in `core/planner/job_cost.py` is fitted on these numbers.
[Three newer model-specific fits](model-cost-validation.md) are used for Qwen3-0.6B, Qwen3-1.7B and Llama-3.2-1B-Instruct.

## Setup

- Base model `meta-llama/Llama-3.1-8B-Instruct`, LoRA r 16, alpha 16, dropout 0.05, batch 8, learning rate 1e-4, packing off, context 8192, seed 0.
- Data: a random BANKING77 subset in chat format (system prompt, query, intent label) with 50 validation examples.
- The 50, 1000 and 3-epoch jobs were submitted within 22 seconds of each other. The 200-example job ran alone before them.

## Measurements

Seconds from job creation, taken from the job events.

| Job | Trained tokens | Steps | Datasets processed | Training started | Training | Completed to succeeded | Total | Cost at $0.40 per 1M |
|---|---|---|---|---|---|---|---|---|
| 50 examples, 1 epoch | 3,577 | 7 | 70 | 223 | 134 | 74 | 431 | $0.0014 |
| 200 examples, 1 epoch | 14,364 | 25 | 62 | 224 | 162 | 76 | 462 | $0.0057 |
| 1000 examples, 1 epoch | 72,779 | 125 | 473 | 617 | 338 | 76 | 1031 | $0.0291 |
| 200 examples, 3 epochs | 43,092 | 75 | 311 | 457 | 305 | 77 | 839 | $0.0172 |

`trained_tokens` counts every token of the chat template once per epoch: about 72 tokens per example with the Llama 3.1 template. The cost column assumes billing uses `trained_tokens`; Billing > Usage had not caught up with these jobs at the time of writing.

## Duration model

- Training: 122 s + 2.97 s per 1000 trained tokens (about 337 tokens/s), plus about 28 s per extra epoch, probably for saving its checkpoint. The three 1-epoch jobs fit within 2 s.
- Fixed stages: about 65 s to process the datasets, 150 s more until "Training started", and 75 s from "Training completed" to status `succeeded`.
- A single job therefore takes at least about 7 minutes. 1000 examples add only 10 minutes over 50.

## What the model leaves out

- **Waiting for capacity.** Earlier the same day three jobs waited 35 minutes to 2 hours with 0 steps. Nebius Support explained that the cluster was busy. A waiting job emits "Training started" like a running one, and `trained_steps`, `total_steps` and `estimated_finish` stay empty. Once the job gets a GPU they fill in. Waiting is not billed.
- **Parallel jobs.** All three parallel jobs trained without waiting for capacity, and two of them trained at the same time. Dataset processing was the slow part: 70, 311 and 473 s, as if the jobs were processed one after another.
- **Cancellation** waits for a scheduling slot too: a cancel of a waiting job took effect 33 minutes later.
- **Other models.** Durations are fitted on Llama-3.1-8B only.

## Other findings

- `result_files` hold a PEFT LoRA adapter (`adapter_model.safetensors`, 84 MB for this model, and `adapter_config.json`), plus the tokenizer and the chat template, not merged weights.
- The adapter config shows PEFT 0.19.1 with LoRA on all seven linear projections (`q/k/v/o_proj`, `gate/up/down_proj`), no rsLoRA or DoRA. The local simulation script `jobs/kaggle/train_lora.py` uses the same setup.
- There is one checkpoint per epoch, and its metrics are only `train_loss` and `valid_loss`. A Token Factory loss curve is one point per epoch.
- The learning rate decays over the whole run. The first-epoch checkpoint of the 3-epoch job differs from the 1-epoch job trained on the same data and seed: valid loss 0.454 against 0.805.
- The 3-epoch job overfits, which makes it a real case of fault 7. Train loss goes 1.169, 0.329, 0.084 while valid loss goes 0.454, 1.032, 2.033.
