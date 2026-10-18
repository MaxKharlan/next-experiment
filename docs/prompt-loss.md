# Mode B fault 9: loss on prompt tokens

The local LoRA trainer accepts `--code-fault prompt_loss`. On a copy of encoded,
unpadded training rows, it replaces ignored prompt labels with the input token
IDs. Answer labels remain unchanged. The collator still ignores appended padding,
and the loss function still shifts predictions once. Position zero therefore
contributes no target. Validation keeps the healthy answer-only labels.

`audit_prompt_labels` compares observed labels with the healthy answer-only
reference after context truncation, before padding. Its contract assumes ignored
positions in these unpadded reference rows are prompt positions. It rejects
misaligned rows, rewritten answer labels and prompt labels that differ from
input tokens. It counts prompt positions, supervised prompt tokens, answer
targets and total targets. `prompt_loss_fraction` is the fraction of targets
belonging to the prompt, or null for an empty objective. `no_prompt_targets`
means the reference has no eligible prompt positions; the trainer refuses a
fault injection that adds no prompt supervision.

The report is saved as `prompt_label_audit` in `summary.json`. It is null for
`eos_as_pad`, whose removal audit remains in `label_audit`. The EOS audit's
explicit `allow_prompt_targets` option accepts restored prompt token IDs while
continuing to reject other label rewrites. Reference EOS supervision stays
separate from any EOS tokens newly supervised in the prompt.

Training metrics and summaries now contain `supervised_tokens`, the cumulative
number of targets used by the loss. `answer_tokens` continues to count answer
targets. For `prompt_loss`, training loss and token accuracy cover both prompt
and answer targets; validation metrics cover answers only. These objectives must
not be compared as though they measured the same tokens.

For a future local experiment with prepared chat JSONL:

```bash
uv run python -m jobs.kaggle.train_lora \
  --train artifacts/train.jsonl --valid artifacts/valid.jsonl \
  --out artifacts/prompt-loss --device cpu --code-fault prompt_loss
```

This command may download weights and train a model. It was not run for this
change. The existing trainer supports local CPU/MPS execution despite its
historical `jobs/kaggle` directory name.

## Verification

```bash
uv run pytest -q tests/unit/test_label_faults.py tests/unit/test_train_lora.py
```

Synthetic CPU tests verify that the injected objective changes the gradient,
while full batches and accumulated microbatches agree within the injected arm.
The trainer integration fixture has 23 answer targets and gains 85 prompt
targets: `supervised_tokens` becomes 108 while `answer_tokens` stays 23.
Validation EOS labels remain intact and appended padding remains ignored.
Other controls cover healthy masking, position zero, input mutation, malformed
labels and refusal of an ineffective injection before model loading.

This implements fault 9 and its label audit within B24. It does not establish
that a trained model repeats prompts or loses downstream quality. Fault 11,
agent-facing checks and downstream quality experiments remain open.
