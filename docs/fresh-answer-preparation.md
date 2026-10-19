# Fresh answer-cut preparation, 19 October 2026

The new JSON extraction cohort has 432 training rows and 108 rows each for
validation, intervention checks and final test. No model has scored this cohort;
no training or paid inference was submitted. The execution plan remains
`prepared_not_executable`. This preparation does not close the full API trace or
the three-task/three-model matrix.

The [review](results/fresh-answer-preparation-20261019/review.json) checks every
reference with the existing text-only parser, separately from the new generator.
The same Codex session authored the generator and ran the review; there was no
independent human or second-agent review. IDs, normalized prompts and semantic
slot groups do not cross splits. Groups combine product, priority, callback and
due-date bucket. Normalization removes ticket IDs and concrete dates. Compared
with the eight historical structured JSONL files, there are no reused IDs or
normalized prompts. Schema and bounded vocabulary are shared with historical
data; semantic overlap with historical tasks is expected. This evaluates new
slot combinations, not broad natural-language extraction.

Clean references and injection line manifests stay in the local evaluator-only
folder. The observed mild/strong folders contain damaged train/validation and
unchanged check/test copies. Only observed train/validation enter agent context;
check/test remain evaluator inputs. Directory naming is a workflow boundary,
not an OS sandbox. All four split hashes are recorded in the
[plan](results/fresh-answer-preparation-20261019/execution-plan.json).

## JSON probe defect

The old rule required 90% of answers to parse before examining broken JSON.
Once damage exceeded 10%, this disabled the JSON rule. Unique entity answers
could then escape the prefix detector too. `answer-cut-v2` instead requires 90%
of answers to open as objects/arrays and a strict majority to parse. It flags
non-parsing object/array answers within that cohort. Brace-prefixed prose without
a parseable majority does not enable the JSON rule. Majority-invalid cohorts
remain outside this heuristic. Malformed JSON is an observed defect; it is not
proof of the unique corruption mechanism.

Tests cover 10%, 30%, 49%, the 50% boundary, clean labels and brace-prefixed prose.
On the prepared cohort, the probe flags all 43 mild and 130 strong damaged
training rows and zero healthy rows. This is a synthetic detector check, not a
model-quality measurement. Existing saved reports retain their old semantics;
new raw results record the version and thresholds. Signature likelihoods remain
heuristic and have not been recalibrated for v2.

## Tokens, budget and execution gates

Only the cached tokenizer was loaded, with PyTorch disabled. Snapshot:
`c1899de289a04d12100db370d81485cdf75e47ca`. The plan records tokenizer, template,
evaluator, prompt and signature hashes. Provider weights are not assumed to
match that local snapshot, and the live evaluator still needs to enforce the
recorded revision before execution.

The strong target and control each retain 302 rows, containing 89,710 and 86,883
rendered tokens respectively. Here the target has 3.25% more tokens. The earlier
assumption that control must cost more is false for this cohort; the data-pair
docstring now states the measured resource boundary.

The agent selected a $5 cap within the user's permission for necessary paid
experiments. A conservative local token estimate for at most 26 jobs is $1.375
at the repository's $0.40/million rate, plus $1 inference reserve and $0.50
contingency: $2.875 estimated total. This is not a provider quote or invoice.
Price verification is required before submission. The plan reserves six
development calibration jobs, three fresh references, up to eight seeds per
pair arm and one final retrain.

Before any fresh model evaluation, measure development noise on the historical
structured cohort. Minimum useful effect is preselected at 0.08; simulated power
is assessed at an anticipated 0.16 effect. Choose 3–8 seeds using the full decision
rule and a lower Monte Carlo power bound of 0.80. Stop if no candidate passes,
weights are identical or the variance estimate is zero. Do not import BANKING77's
0.019 SD. These are planning assumptions, not measured effects.

The run stops after one strong-damage API investigation and at most one approved
final retrain. No adaptive extra seeds or final-test-driven recipe changes are
allowed. Budget exhaustion, changed hashes and ambiguous provider submission
stop execution. The approved execution protocol and its hash must be frozen after
development calibration. Reference jobs have one replicate each, so this design
does not estimate reference variance or close a matrix cell. A matched-compute
sensitivity is still needed before claiming specificity where compute imbalance
could explain the result.

B9 reported 144/210 completed runs on 19 October during this preparation. It owns
the active MPS process. Its agreed order is to finish all 210, release MPS, then
perform CPU analysis and transfer scoring while this chat evaluates adapters.
This dated checkpoint is not a live progress counter.

## Reproduce offline

From the repository root, choose a new output directory:

```sh
uv run python -m bench.tasks.fresh_extraction \
  --out artifacts/fresh-extraction-reproduction

USE_TORCH=0 USE_TF=0 uv run --no-project \
  --with 'transformers==5.19.0' --with 'pydantic==2.14.1' \
  --with 'jinja2==3.1.6' python -m scripts.prepare_fresh_answer \
  --cohort artifacts/fresh-extraction-reproduction \
  --snapshot "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca" \
  --out artifacts/fresh-extraction-reproduction/execution-plan.json
```

The second command requires that cached snapshot and records resolved dependency
versions. It never downloads model weights or submits provider jobs. Output paths
must be new. The plan builder rejects data changes after the recorded review.
