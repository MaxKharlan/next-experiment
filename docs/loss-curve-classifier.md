# B9: first loss-curve classifier

`bench.loss_classifier` extracts five features from saved local loss traces:
training and validation slopes, the final validation-minus-training gap,
the normalized position of minimum validation loss, and training-loss change
in the last 20% of steps. Each curve uses its own normalized step coordinate.
With fewer than two tail points, the last two training points are used.
Hyperparameters, exact-match scores and run names are not classifier inputs.

The baseline standardizes features and assigns the nearest class centroid by
squared Euclidean distance. Ties use alphabetical label order. Constant features
use scale one. It outputs recipe labels, not calibrated probabilities.

## Evaluation on saved runs

The existing sweep contains 15 Qwen3-0.6B runs on 500 BANKING77 training examples
and 100 validation examples: healthy, low learning rate at two strengths, and
small LoRA rank at two strengths; each recipe has seeds 0, 1 and 2.
No new fine-tuning or API calls were made.

Evaluation leaves out an entire seed across every recipe. Scaling and centroids
are fitted only on the other seeds. Each run receives exactly one held-out
prediction. Recipes, task and model are shared across folds; this measures
transfer to another seed, not to another recipe family, task or provider.

| Actual recipe | Predicted healthy | Predicted small rank | Predicted low LR |
|---|---:|---:|---:|
| Healthy | 3 | 0 | 0 |
| Small rank | 6 | 0 | 0 |
| Low LR | 2 | 0 | 4 |

Accuracy is 7/15 (46.7%); balanced accuracy is 55.6%. Predicting one of the two
largest classes for every run would yield 6/15 (40%). The baseline misses every
small-rank run. No feature or hyperparameter search was performed against these
fold results.

The [evaluation](results/loss-classifier-local-20261018/evaluation.json) contains
predictions, fold membership, each fold's scaler and centroids, and the code hash.
The [feature table](results/loss-classifier-local-20261018/rows.json) includes
hashes of original summaries and loss traces. Those numeric inputs are bundled
under `inputs/`. The [model](results/loss-classifier-local-20261018/model.json)
is fitted on all 15 runs for later exploratory transfer tests; it was not used
to calculate the cross-validation scores.

## Reproduce

```bash
uv run python -m bench.loss_classifier \
  --runs docs/results/loss-classifier-local-20261018/inputs \
  --out artifacts/b9/loss-classifier-reproduction
```

Use a new output directory. The command rejects duplicate recipe/seed identities,
mixed base models, invalid losses, insufficient curves and folds missing a
training class. It copies the original numeric inputs into its output bundle.

## What remains in B9

These labels identify simulated recipe changes. Small rank did not reduce exact
match in this sweep, so recognizing it would not establish a real failure.
There are no excess-epoch runs in this bundle. A deployable diagnostic classifier
still needs more runs, tasks with verified quality degradation, an evaluation
across recipe strengths, and transfer testing on Token Factory curves. Token
Factory has per-epoch losses, while these local traces have per-step training
losses; that sampling difference must be assessed before reuse.

The model is experimental and is not connected to the live agent or published
as a validated classifier. This delivers the first training/evaluation pipeline
for B9; it does not close the complete simulation and transfer requirement.
