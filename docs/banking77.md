# A1: BANKING77 splits

Build the generative intent-label task without model calls or training:

```bash
uv run python -m bench.tasks.banking77 --out artifacts/tasks/banking77
```

The builder downloads the original PolyAI BANKING77 CSV files from a pinned Git
revision. The dataset is licensed under CC BY 4.0; the source repository, revision
and source file hashes are recorded in `manifest.json`.

Each JSONL record has a system instruction, the customer query as a user message,
and the intent label as the assistant answer. This uses the same instruction as
the existing B6 preparation script. The metric is exact match of the intent label.

| File | Source and purpose |
| --- | --- |
| `train.jsonl` | Remaining official training examples, for fine-tuning |
| `validation.jsonl` | 10 examples per intent from official train, for training validation |
| `check.jsonl` | Another 10 examples per intent from official train, for causal checks |
| `test.jsonl` | Official test pool, reserved for the final model comparison |

The default seed is 0. Use `--seed`, `--validation-per-label` and
`--check-per-label` to build a different configuration in a new output directory.
Every intent must retain at least one training example. Sampling is stratified
by intent; test membership does not depend on the seed.

Queries are compared after Unicode NFKC normalization, case folding and whitespace
normalization. Duplicate queries are removed before splitting, with official test
queries taking precedence. Conflicting labels stop the build. This excludes exact
normalized overlap, not semantic near-duplicates; those belong to A10.

The manifest includes the seed, split sizes, duplicate counts and SHA-256 hashes.
Its `datasets` field follows `DatasetSplits`; resolve each `ref` relative to the
manifest directory before passing it to an investigation. Output directories must
be new, so a build cannot overwrite previous experiment inputs.

Generated files stay under the ignored `artifacts/` directory. Existing B6/B10/B18
files, saved demo runs and benchmark results are not migrated by this command.
These new splits do not retroactively change or validate the published benchmark.
Do not tune thresholds on `test.jsonl`.
