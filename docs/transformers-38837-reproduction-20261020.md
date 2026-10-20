# Transformers #38837: partial CPU mechanism reproduction

The historical Trainer reproduces the half-sized final gradient with accumulation
2. A local divisor correction restores the analytic SGD result in that setting.
The broader prespecified experiment **stopped**: accumulation 3 disagrees with the
analytic reference in both original and corrected arms. This package provides
partial mechanism evidence, with no pretrained quality or original-user-run claim.

## Source and scope

[Transformers #38837](https://github.com/huggingface/transformers/issues/38837)
reports final-group loss scaling in Transformers 4.51.3. The maintainer distinguishes
the mean-loss path from models that accept loss kwargs and count batch items.
This reproduction exercises the mean-loss path on CPU with a small linear model.
It does not reproduce the reporter's BERT weights, GPU environment or downstream
quality, and does not establish a current Token Factory defect.

The isolated environment pins PyTorch 2.6.0, Transformers 4.51.3 and Accelerate 1.6.0.
It uses Python 3.11.15 on macOS; the reported original used Python 3.10.16 on Linux.
The [requirements](results/transformers-38837-cpu-v2/requirements.txt) record all
installed dependencies. No existing runtime was modified or model downloaded.

## Frozen experiment

A float64 linear model starts with weights [0.25, 0.5] and bias 0.125. Every
synthetic example has x=[1,2], y=1. SGD has no momentum, clipping or weight decay.
The microbatch size is 2; the cases are 8 examples/accumulation 2,
10 examples/accumulation 2 and 10 examples/accumulation 3. Each runs at learning
rates 0 and 0.05, using original and corrected Trainer arms: 12 CPU runs.
An independent algebraic calculation supplies expected gradients and parameters.
The absolute tolerance is 1e-10; all cases must agree for the planned pass.

The correction records the actual group size returned by `get_batch_samples`,
uses that divisor inside `training_step`, and restores the configured accumulation
count before returning. It is a bounded experimental patch, not a deployment fix.

## Results

At zero learning rate, the full analytic gradient is [0.75, 1.5, 0.75].

| Examples / accumulation | Original final gradient | Corrected final gradient | Prespecified comparison |
| --- | --- | --- | --- |
| 8 / 2 | [0.75, 1.5, 0.75] | [0.75, 1.5, 0.75] | Both agree |
| 10 / 2 | [0.375, 0.75, 0.375] | [0.75, 1.5, 0.75] | Expected half-gradient and correction agree |
| 10 / 3 | [1, 2, 1] | [1.5, 3, 1.5] | Both fail the planned reference |

At learning rate 0.05, the 10/2 final parameters are
[0.1945, 0.389, 0.0695] in the original arm and
[0.1915, 0.383, 0.0665] in the corrected arm. The latter agrees with full-batch
SGD; the divisor changes the actual parameter update.
Four of six case/rate combinations match their planned references.
The two accumulation-3 combinations fail. Their observed outputs remain in the
[raw evidence](results/transformers-38837-cpu-v2/failed-results.json).

Inspection of the installed historical Trainer shows separate remainder and
synchronization calculations: one uses example count, while the loop operates
on microbatches. The accumulation-3 run extended to epoch 1.6 in its console log.
These observations motivate further diagnosis, but this experiment does not
isolate every scheduling effect. No third run or revised tolerance was selected.

## Failed preflight retained

The [first protocol](results/transformers-38837-cpu-preflight-v1/protocol.json)
used one epoch with default `max_steps`. The historical Trainer stopped at epoch
0.8 for 10/2 and 0.6 for 10/3, so it did not reach their remainder groups. The
update-count gate failed. Its first runner checked results before saving;
failed-run gradients were therefore not retained. The failure record states
this limitation and does not reconstruct the missing values.

Revision 2 explicitly set `max_steps=ceil(microbatches/accumulation)` to request
a complete data pass and saved failures before raising. Cases, model, learning
rates and tolerance stayed fixed. Revision 2 still failed its full criterion.
Together the two revisions executed 24 tiny CPU Trainer runs, with zero provider
calls or charges. Neither revision changes the completed B9/B12 studies.

## Verification and reproduction

The [analysis](results/transformers-38837-analysis-20261020/report.json) checks
manifest coverage, protocol/source bindings, all saved gradients and parameter
updates against the algebraic reference. It preserves `stopped` and the failed
controls. Verify without torch or transformers:

```bash
python -m scripts.report_accumulation_remainder \
  --root docs/results/transformers-38837-cpu-v2 \
  --out docs/results/transformers-38837-analysis-20261020/report.json --verify
```

For an intentional new software reproduction, create an isolated Python 3.11
environment using the saved requirements. With that environment's Python:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.reproduce_accumulation_remainder \
  prepare --out /new/output/directory
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.reproduce_accumulation_remainder \
  run --out /new/output/directory
```

The run is expected to exit nonzero at the frozen mismatch and retain
`failed-results.json`. Existing results or failures prevent rerunning that output
directory. The exact runner source is included with each frozen revision.

B21 remains partial. This evidence covers one adapted training mechanism, and the
full prespecified reproduction failed. Original-case reproduction, downstream
quality and the other collected public cases remain open. No benchmark truth
label or live classifier/signature was added.
