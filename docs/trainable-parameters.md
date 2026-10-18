# B24: trainable-parameter inspection

The local LoRA trainer audits parameters after attaching PEFT and moving the
model to its device, before creating the optimizer or running evaluation. It
writes `parameter-audit.json` in the output directory immediately and includes
the same report under `parameter_audit` in a completed `summary.json`.

The report contains total, frozen and trainable parameter counts, the trainable
fraction, LoRA counts, and names, shapes and sizes of trainable tensors. Counts
use the model's standard `named_parameters()` deduplication, so tied tensors are
counted once. It stores metadata only, not parameter values.

Exact `lora_A` and `lora_B` path components identify the linear LoRA parameters
used by this trainer's PEFT configuration. Other trainable tensors are reported
separately. Findings are:

- `no_trainable_parameters`: no scalar parameters are trainable. The trainer
  saves the audit and raises before constructing an optimizer.
- `no_trainable_lora_parameters`: no trainable linear LoRA A/B weights were found.
- `trainable_non_lora_parameters`: weights outside the linear LoRA A/B modules
  are trainable, for example an accidentally unfrozen base embedding.

The latter two findings remain visible in the report without aborting training.
The audit never changes `requires_grad`. Its classification is scoped to this
trainer's linear LoRA setup with `bias=none`; other adapter types, trainable
biases and intentionally saved trainable modules need a different expectation.
This snapshot does not establish that gradients reach a parameter or that the
optimizer updates it, and it does not monitor changes during training.

## Verification

```bash
uv run pytest -q tests/unit/test_trainable_parameters.py tests/unit/test_train_lora.py
```

A GPT-2 model built locally from a tiny configuration receives a real PEFT LoRA
adapter, without downloading weights or using Kaggle. Its adapter has 64
trainable parameters and a frozen base. Deliberately unfreezing its embedding
reports 128 trainable non-LoRA parameters. Freezing the entire model reports
zero trainable parameters while retaining the adapter inventory.

Other tests cover tied weights, empty models, exact component matching and
unchanged freeze state. Trainer integration tests compare the saved audit with
the summary and verify that a fully frozen model saves its report before
stopping, without creating an optimizer or metrics log. Existing CPU training
fixtures still run through all label fault modes.

This completes the trainable-parameter inspection portion of B24. Agent-facing
checks, gradient-flow verification and downstream quality experiments remain
open; the report does not assign benchmark ground truth.
