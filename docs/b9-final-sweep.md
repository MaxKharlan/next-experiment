# B9: 210-run loss-curve sweep

The completed sweep contains 210 local Qwen3-0.6B LoRA trainings: seven recipes
at 30 seeds. Each run uses 50 BANKING77 training rows and 16 validation rows.
The sweep finished with exit code 0. All expected `summary.json` and
`metrics.jsonl` files parsed, and the matrix has no missing or duplicate
recipe/seed cells.

## Downstream quality

The healthy recipe averages 0.1292 exact match. Both low-learning-rate recipes
score 0 on every seed. Small ranks do not lower exact match. Three and six
epochs improve it on this small dataset.

| Recipe | Exact match, mean ± SD | Paired difference from healthy | 95% paired bootstrap interval |
| --- | ---: | ---: | ---: |
| Healthy, 1 epoch, rank 16 | 0.1292 ± 0.0432 | — | — |
| Learning rate ×0.1 | 0.0000 ± 0.0000 | -0.1292 | [-0.1437, -0.1146] |
| Learning rate ×0.03 | 0.0000 ± 0.0000 | -0.1292 | [-0.1437, -0.1146] |
| 3 epochs | 0.2313 ± 0.0291 | +0.1021 | [+0.0833, +0.1208] |
| 6 epochs | 0.1833 ± 0.0517 | +0.0542 | [+0.0312, +0.0771] |
| Rank 4 | 0.1313 ± 0.0445 | +0.0021 | [-0.0042, +0.0104] |
| Rank 2 | 0.1354 ± 0.0437 | +0.0063 | [0.0000, +0.0125] |

The bootstrap resamples paired seeds 20,000 times with seed 20261019. Low
learning rate loses to healthy on all 30 seeds. Rank 4 ties healthy on 27 seeds;
rank 2 also ties on 27. The epoch recipes improve on 28/30 and 19/30 seeds.

These outcomes support one failure class in this matrix: low learning rate.
The rank and epoch settings are planted recipe changes, but they are not quality
failures here.

## Loss-curve classification

The feature extractor uses training slope, validation slope, final train/valid
gap, normalized position of minimum validation loss and late training-loss
change. Exact match, recipe names and hyperparameters are excluded from model
inputs. Scaling and centroids are fitted inside every evaluation fold.

| Target and split | Accuracy | Balanced accuracy |
| --- | ---: | ---: |
| Four planted fault families, leave one seed out | 71.4% | 63.8% |
| Seven recipe/strength cells, leave one seed out | 56.7% | 56.7% |
| Four families, unseen strength and disjoint seed half | 59.6% | 59.6% |
| Observed low-LR failure versus no observed drop, leave one seed out | 98.1% | 96.7% |

The four-family model identifies 57/60 low-LR curves and 48/60 excess-epoch
curves. It labels only 3/30 healthy curves as healthy. The distinction between
healthy and small-rank curves is not useful: their mean losses and exact-match
scores are almost identical.

The binary model treats healthy, small-rank and excess-epoch runs as the
no-observed-drop class. It correctly classifies all 150 such local runs and
56/60 low-LR runs under leave-one-seed-out evaluation. This number describes the
local matrix only.

## Frozen Token Factory transfer

The binary model was fitted once on all 210 local curves, then applied without
refitting to the 13 saved Token Factory cases. Those cases were inspected in
earlier project work, so this is a historical transfer test.

| Split | Accuracy | Balanced accuracy | Low-LR recall | Healthy specificity |
| --- | ---: | ---: | ---: | ---: |
| Dev, 5 cases | 40.0% | 41.7% | 1/3 | 1/2 |
| Historical heldout, 8 cases | 12.5% | 25.0% | 0/6 | 1/2 |

The local signature does not transfer to Token Factory. Per-step local traces
and per-epoch provider checkpoints have different sampling and loss behavior.
The model stays out of live diagnosis, and B26 publication remains blocked.

## Artifacts and reproduction

The result directory `docs/results/b9-sweep-50-20261019/` contains:

- `report.json`: outcome aggregates, bootstrap settings, four classifier
  evaluations and SHA-256 hashes for all 420 numeric input files;
- `recipe-model.json`: the four-family centroid model fitted on all local runs;
- `observed-failure-model.json`: the binary model used for frozen transfer;
- `token-factory-transfer.json`: predictions, case hashes and split metrics;
- `validation.txt`: commands, test counts and hashes from the deterministic rerun.

Reproduce the analysis from a completed sweep without loading model weights:

```bash
uv run python -m bench.sweep_analysis \
  --runs artifacts/next-batch/sweep-50 \
  --out artifacts/b9-final-analysis
```

The command requires all seven recipes at seeds 0–29. It rejects missing,
unexpected and duplicate cells. The result records hashes of the analysis code,
summaries, loss traces, model artifact and Token Factory cases.

The sweep used no provider training service. The B12 heldout collection in the
same project package used eight Super calls costing $0.0135807. No additional
provider calls were made for this analysis.
