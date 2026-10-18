# B24: first-backward gradient audit

The local LoRA trainer audits trainable tensors after backward for the first
optimizer group, before gradient clipping and before the optimizer step. With
accumulation, the snapshot includes all microbatches in that group. It writes
`gradient-audit.json` immediately and includes the same report as `gradient_audit`
in a completed `summary.json`.

Each unique trainable tensor receives one status:

- `missing`: `.grad` is absent; this is distinct from a zero-valued gradient.
- `zero`: a gradient exists, all its values are finite and zero.
- `nonzero`: all values are finite and at least one is nonzero.
- `nonfinite`: at least one value is NaN or infinity.

Sparse COO gradients are inspected through coalesced values. The report contains
names, tensor sizes, status counts and findings; it contains no gradient values.
The audit does not mutate gradients, freeze state or optimizer state. Frozen
parameters are outside its scope.

The trainer saves the report and stops before clipping or updating weights when
all gradients are missing, no trainable tensors exist, or any gradient is
nonfinite. Some missing gradients and all-zero gradients are recorded without
aborting. These observations can be legitimate for conditional paths or a
particular batch. In standard LoRA initialization, B starts at zero, so A can
receive a zero gradient on the first backward while B receives a nonzero gradient.

This is a snapshot of the first successful backward call. It does not monitor
later steps, test that the optimizer changes parameters, or diagnose exceptions
raised inside backward itself. Gradient clipping and its numerical behavior are
outside this snapshot. No extra forward/backward pass is introduced.

## Verification

```bash
uv run pytest -q tests/unit/test_gradient_audit.py tests/unit/test_train_lora.py
```

A real PEFT adapter on a tiny GPT-2 model created locally receives a backward
pass: LoRA A has a zero gradient and B has a nonzero gradient, with no findings.
Other tests distinguish absent, zero, nonzero and nonfinite gradients, cover a
sparse embedding and check that inspection preserves tensors. Trainer tests
verify the saved summary/report, including accumulated batches, and inject
missing or NaN gradients to verify that the report is saved before an optimizer
step can run. No downloaded weights, Kaggle jobs or paid training were used.

This implements gradient-flow inspection within B24. Agent-facing checks and
downstream quality evidence remain open.
