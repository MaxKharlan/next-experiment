# Mode B fault 11: labels shifted twice

The local LoRA trainer accepts `--code-fault double_shift`. It pre-shifts each
encoded, unpadded training label sequence one position left and appends an ignored
label (`-100`). The existing causal loss then shifts once more. Prediction at
position `t` is therefore trained against the original label at `t + 2` instead
of `t + 1`. Input tokens and attention masks remain unchanged. There is no wrap
at the last token or across rows, and appended padding remains ignored.

For example, healthy labels `[-100, -100, 3, 4]` become `[-100, 3, 4, -100]`.
After the causal loss shift, the effective targets are `[3, 4, -100]`, compared
with healthy targets `[-100, 3, 4]`. A target moved to position zero is lost
because there is no preceding prediction. The trainer rejects rows that lose
all supervision and refuses injections that do not change effective alignment,
before loading model weights. Validation keeps the original answer-only labels.

`audit_label_shift` compares the observed effective targets with two references:
healthy labels and labels pre-shifted once. It saves mismatch counts, informative
positions and the number of observed targets. Its statuses are `aligned`,
`double_shift`, `other_mismatch`, and `inconclusive`. Identical reference patterns
provide no evidence to distinguish alignment. The result describes supplied
label tensors, not downstream model quality or a general diagnosis from loss.
Input rows and token sequences must align exactly with the healthy reference.

The trainer saves the result in `label_shift_audit` in `summary.json` for healthy
runs, accumulation fault runs and double-shift runs. It is null for other label
faults. For double-shift runs, `label_audit` (EOS removal) and `prompt_label_audit`
are null because their masking assumptions do not apply to moved labels.

For a future local experiment with prepared chat JSONL:

```bash
uv run python -m jobs.kaggle.train_lora \
  --train artifacts/train.jsonl --valid artifacts/valid.jsonl \
  --out artifacts/double-shift --device cpu --code-fault double_shift
```

This command may download weights and train a model. It was not run for this
change. The existing trainer runs on local CPU/MPS despite the historical
`jobs/kaggle` directory name.

## Verification

```bash
uv run pytest -q tests/unit/test_label_faults.py tests/unit/test_train_lora.py
```

Synthetic CPU tests check exact shifted targets, ignored padding, nonmutation,
alignment errors, empty and one-token rows, and early rejection of ineffective
or empty objectives. A deterministic next-token model has near-zero loss and
3/3 correct targets with healthy labels; double shifting gives loss above 19
and 0/2 correct targets. These are constructed logits, not measured LLM quality.

The trainer integration test checks the recorded audit and agreement between
full-batch and accumulated AdamW updates, including the final incomplete group.
Validation EOS labels remain unchanged. No downloaded model or paid training
was used. B24 now has injection mechanisms for faults 9–12;
[trainable-parameter inspection](trainable-parameters.md) is now implemented;
agent integration and downstream quality evidence remain open.
