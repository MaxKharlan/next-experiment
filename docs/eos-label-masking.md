# Mode B fault 10: EOS masked as padding

The local LoRA trainer accepts `--code-fault eos_as_pad`. It sets the padding ID
to the tokenizer's EOS ID and masks every matching token position in training
labels. This deliberately removes genuine EOS targets along with padding. The
existing collator normally masks only appended padding positions, so sharing a
padding and EOS ID alone does not cause this fault.

`bench.faults.labels.audit_eos_labels` compares healthy and observed labels after
context truncation. It excludes the first position (no preceding prediction) and
already ignored prompt labels. The report counts supervised targets before and
after masking, reference EOS targets, and removed EOS targets. The removal
fraction is null when there are no reference EOS targets; that status is
`no_eos_targets`, not evidence that masking is correct. The function requires
aligned input tokens and accepts only label masking by default. Its explicit
`allow_prompt_targets` option supports the [prompt-loss audit](prompt-loss.md);
other label rewrites are rejected.

The trainer saves this report as `label_audit` in `summary.json`, alongside the
selected `code_fault`. For tokenizers without an EOS ID, healthy runs record a
null audit. Fault injection requires an EOS ID and at least one removed EOS
target. Training rows left without any supervised next-token targets are rejected
before loading model weights. Validation labels keep the healthy answer-only
objective, including EOS.

For a future local experiment with prepared chat JSONL:

```bash
uv run python -m jobs.kaggle.train_lora \
  --train artifacts/train.jsonl --valid artifacts/valid.jsonl \
  --out artifacts/eos-as-pad --device cpu --code-fault eos_as_pad
```

The directory name `jobs/kaggle` is historical; this command runs on the selected
local device. It may download model weights and train a model. It was not run for
this change. Faults are selected one at a time through `--code-fault`; recipe
faults remain a separate option.

## Verification

```bash
uv run pytest -q tests/unit/test_label_faults.py tests/unit/test_train_lora.py
```

Tests use synthetic tokens and a tiny deterministic CPU model without downloads.
The training integration test removes all five supervised EOS targets while
retaining 18 answer targets. Healthy training retains all 23 targets. Both full
batches and accumulated microbatches produce matching updates within each arm;
validation retains its EOS labels in both arms. Controls cover distinct padding,
healthy shared padding, ignored prompt EOS, an EOS at position zero, partial
removal and absent EOS supervision. Invalid fault inputs fail before model loading.

This implements the fault-10 masking mechanism and its label audit within B24.
It does not measure EOS generation or downstream quality, reproduce the original
TRL #696 run, or assign benchmark ground truth. Chat templates that end with a
separate end-of-turn token need an audit of that token before attributing stopping
behavior to EOS. Agent-facing checks and the remaining B24 mechanisms stay open.
