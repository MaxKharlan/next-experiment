# Mode B fault 12: microbatch loss weighting

The local LoRA trainer (`jobs/kaggle/train_lora.py`, also usable on CPU and MPS)
now supports `gradient_accumulation_steps` in its hyperparameter JSON. The default
is one, preserving the existing optimizer-step schedule. `batch_size` remains
the number of examples in one microbatch. Clipping, AdamW and the learning-rate
scheduler run once per accumulation group, including the final incomplete group.

The healthy objective gives each supervised next-token target equal weight. For
microbatch mean loss `L_i`, supervised target count `n_i`, and total count `N` in
the current group, backward uses `L_i * n_i / N`. Counts exclude ignored labels
and the first position, which has no preceding prediction. Padding contributes
no targets. An empty group or a microbatch without targets raises before any
backward call in that group.

`--code-fault microbatch_mean` deliberately uses `L_i / K`, where `K` is the
actual number of microbatches in the group. With unequal target counts this
changes the gradient. It implements fault 12 from the plan; it does not implement
the separate historical bug of dividing a remainder group by the configured
accumulation count. This option requires accumulation greater than one and is
recorded separately from recipe faults in `summary.json`.

For a future local training experiment with prepared chat JSONL:

```bash
uv run python -m jobs.kaggle.train_lora \
  --train artifacts/train.jsonl --valid artifacts/valid.jsonl \
  --out artifacts/accumulation-healthy --device cpu \
  --hp '{"batch_size": 1, "gradient_accumulation_steps": 3}'

uv run python -m jobs.kaggle.train_lora \
  --train artifacts/train.jsonl --valid artifacts/valid.jsonl \
  --out artifacts/accumulation-fault12 --device cpu \
  --hp '{"batch_size": 1, "gradient_accumulation_steps": 3}' \
  --code-fault microbatch_mean
```

These commands can download a model and run training; they were not executed for
this change. Validation used a deterministic tiny CPU model with synthetic token
sequences, no network access or downloaded weights:

```bash
uv run pytest -q tests/unit/test_train_lora.py
```

The checks compare gradients, objective values and target counts against a full
batch for equal and unequal answer lengths, a single microbatch, and a remainder
group. The injected fault differs for unequal lengths and agrees for equal
lengths. A trainer integration test compares the parameters after two AdamW
steps with batch size 3 versus batch size 1 accumulated three times, using five
examples so the last group is incomplete. Dropout is absent from this test;
real stochastic training is not expected to be bitwise identical across batch
layouts. Separate tests reject invalid accumulation settings before model loading.

This implements one B24 mechanism and its regression checks. It provides no
measurement of downstream model-quality degradation, no agent-facing diagnosis,
and no reproduction of an external user's training run. Fault 10 now has a [label-masking implementation and audit](eos-label-masking.md).
Faults 9 and 11 remain open. Validation data still uses the healthy answer-only objective
in both experimental arms.
