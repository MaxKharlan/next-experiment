# Mode B local audit and repair package

This package adds three local capabilities to B24. It does not close B24's
broader LLM quality and deployment work. Selection was checked against main
`e10f4b5` before the user authorized three B24 sub-tasks.

## Label audits through REST and the screen

`POST /api/investigations/label-audit` accepts an optional title and the existing
bounded `training_labels` schema. It recomputes alignment, prompt-supervision
and EOS-mask audits. It rejects extra fields, mismatched token sequences and
invalid references before creating an investigation. It accepts no file paths
or code. Reference correctness remains the caller's responsibility.

```json
{
  "title": "EOS label check",
  "training_labels": {
    "reference": [{"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, 2]}],
    "observed": [{"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, -100]}],
    "eos_token_id": 2
  }
}
```

The response contains an investigation ID and audit. Its events, downloadable
report and restart recovery use the existing investigation store, including
SQLite. Saved evidence contains counts and an input SHA-256, without token
sequences. No router, provider or training call is made. The diagnosis remains
`inconclusive`; the summary records `audit_only: true`.

Open the API screen and expand **Inspect local training labels · free** to paste
the request. The event card shows all three audit statuses and expandable details.
The public demo hides the form and rejects creation and access to local audit
investigations. Static pages keep the form hidden.

This endpoint inspects labels. Parameter and gradient inventories remain trainer
artifacts; arbitrary trainer-summary uploads are not supported. The endpoint
does not launch a mode-B LLM investigation or an automatic repair.

## Local interventions

`core.interventions.code_faults.make_code_pair` supports the four catalog
classes for faults 9–12. The context must explicitly declare `mode="B"` and
`job_config["code_fault"]` as one of the trainer switches or null for healthy
code. A mismatched hypothesis is rejected. The target switches to the healthy
implementation; the control repeats the declared observed implementation.
Neither arm changes the recipe, seed set or number of optimizer updates.
Supervised-token counts can differ for label faults; equal compute schedule
therefore does not imply equal supervised-token counts or equal wall time.

`LocalCodePairRunner` executes both arms in new directories through a local
training/evaluation callback. It requires distinct seeds, rejects populated arms
and existing output directories, checks finite scores and publishes metrics only
when both arms finish. Interrupted runs remain on disk and require a fresh output
directory; they are not reused as cached evidence.

`LocalLoRAExecutor` connects this runner to `jobs.kaggle.train_lora.train`.
It passes `code_fault` as a separate argument, trains using train/validation
files, then evaluates the saved adapter on a separate check JSONL. It rejects
empty or exactly overlapping rows across the three files and verifies file
hashes before every run. Model loading may download weights if they are absent;
this adapter was tested with mocked model execution and the existing trainer's
CPU tests, without downloading or training a pretrained LLM in this package.

`investigate_code` connects the factory and runner to the existing agent loop,
with one intervention by default. It exposes only classes matching the declared
switch (all four for an explicitly healthy negative control). This is an opt-in
Python entry point. It makes no change to Token Factory recipe builders or the
existing REST paid-investigation path. Example wiring for caller-owned inputs:

```python
from pathlib import Path
from core.interventions.code_faults import LocalCodePairRunner, LocalLoRAExecutor

executor = LocalLoRAExecutor(
    Path("train.jsonl"),
    Path("validation.jsonl"),
    Path("check.jsonl"),
    model_id="Qwen/Qwen3-0.6B",
    device="cpu",
)
runner = LocalCodePairRunner(base_hp, [0, 1, 2], Path("artifacts/code-pair"), executor)
# runner(target, control), or investigate_code(ctx, case, router, runner, ...)
```

These repairs cover the existing explicit fault switches. They do not rewrite
arbitrary user training source or infer which switch implements an unknown bug.

## Measured generation quality

The offline experiment trains a GRU from scratch on 96 synthetic sequences and
scores greedy generation on 48 separate sequences. Prompts contain random tokens
followed by a category marker; answers have category-dependent length and end in
EOS. Every run uses 100 AdamW updates and four microbatches per update. Six seeds
are paired across target and control. The protocol and all five comparisons were
fixed before evaluating test answers; no outcome was used to tune the recipe.

| Declared switch | Control exact match | Target exact match | Decision on this fixture |
| --- | --- | --- | --- |
| Prompt loss | 1.00 | 1.00 | Rejected: no quality gain |
| EOS masked as padding | 0.00 | 1.00 | Confirmed |
| Double label shift | 0.00 | 1.00 | Confirmed |
| Microbatch mean | 1.00 | 1.00 | Rejected: no quality gain |
| Healthy negative control | 1.00 | 1.00 | Rejected: no quality gain |

Each value is identical across all six seeds. The two positive comparisons have
an effect of 1.0 and an exact one-sided permutation p-value of 1/924. With zero
observed seed variance, the existing seed-based interval collapses to [1, 1];
it does not express uncertainty across tasks or test samples. The five comparisons
are reported separately without a family-wide discovery claim. Prompt loss and
microbatch weighting alter training objectives but did not harm generation on
this easy fixture. This result cannot establish robustness on LLM fine-tuning.

The [result bundle](results/b24-local-20261019/results.json) includes the protocol,
input data, hashes, all per-step losses and all generated test predictions.
Weights remain in the local artifact directories and the output ZIP. No external
report is claimed reproduced, and no benchmark truth labels were added.

Reproduce all 60 trainings in a new directory:

```bash
uv run --group ml --group eval python -m scripts.mode_b_quality \
  --out artifacts/b24-quality-reproduction
uv run ruff check .
uv run ruff format --check .
uv run --group ml --group eval pytest -q
```

A second full run with the same protocol produced the same per-seed scores.
Service spending for both runs was USD 0. Local CPU time and electricity are not
priced as provider usage.
