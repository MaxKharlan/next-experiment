# Four trainer faults on Qwen3-0.6B

The completed CUDA cohort contains fifteen LoRA runs: a healthy trainer and four
injected faults, each at seeds 4101–4103. Prompt loss reduced mean check exact
match by 13.02 percentage points relative to the healthy trainer. Double label
shifting scored zero at every seed. Masking EOS as padding and averaging
microbatch losses showed no quality decline in this setting.

| Trainer | Seed 4101 | Seed 4102 | Seed 4103 | Mean exact match | Sample SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Healthy | 85.94% | 87.50% | 87.50% | 86.98% | 0.90 pp |
| Prompt loss | 75.00% | 70.31% | 76.56% | 73.96% | 3.25 pp |
| EOS masked as padding | 85.94% | 87.50% | 87.50% | 86.98% | 0.90 pp |
| Double label shift | 0.00% | 0.00% | 0.00% | 0.00% | 0.00 pp |
| Microbatch mean | 87.50% | 87.50% | 87.50% | 87.50% | 0.00 pp |

The unchanged base model scored 14.06% on the same 64 check examples. Its
predictions were recovered from the earlier export failure and verified before
reuse. They were not regenerated.

These are development comparisons on one fixed check set. The SD describes
training variation across three seeds; it excludes uncertainty from sampling
different check examples. Healthy-minus-prompt-loss differences were 10.94,
17.19 and 10.94 pp. Healthy-minus-microbatch-mean differences were −1.56, 0 and
0 pp. We did not select new fault strengths or recipes after viewing these
results.

## Frozen setup and collection

All arms used the same 96 training queries, 32 validation queries and 64 check
queries from eight BANKING77 intents. The model was Qwen3-0.6B at revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Training used three epochs, learning rate
0.0002, batch size 2, accumulation 4, LoRA rank/alpha 16, dropout 0.05 and a
512-token context. Greedy generation disabled thinking and allowed 32 new tokens.
Every generated answer ended on EOS; none reached the token limit.

The L40S worker used FP32 with TF32 disabled and math SDPA, following the
[numerical qualification](results/cloud-precision-20261020/decision.json).
This CUDA cohort is analyzed separately from the two supplemental MPS adapters.
Their measurements are not pooled.

The code subprocess in job `aijob-e00emdv72kgz55nqqh` exited successfully before
collection. Its data-fault subprocess was still running. We downloaded 169 code
artifacts and checked their manifests, model revision, package versions, recipe
and seeds. Prediction identities and exact-match metrics were recomputed against
the frozen check file. All fifteen expected cells were present.

The [collection receipt](results/cloud-matrices-export-20261020/completed-code-study/collection.json)
records this completed subprocess separately from the parent job.
[Analysis](results/cloud-matrices-export-20261020/completed-code-study/analysis.json)
contains exact scores and paired differences; the
[frozen protocol](results/cloud-matrices-export-20261020/bundle/code/experiment.json)
binds the source, data and runtime. Individual predictions, training summaries
and logit rows are under the receipt's adjacent `measurements` directory.
Cloud reservations remain held pending provider billing.

## Teacher-forced diagnostics

Each adapter was also measured on the same sixteen validation references.
The table averages the three runs' regional mean NLL and EOS probability after
the complete reference answer.

| Trainer | Prompt NLL | Answer NLL | Template-token NLL | EOS probability after reference |
| --- | ---: | ---: | ---: | ---: |
| Healthy | 4.90994 | 0.06407 | 6.16885 | 0.999870 |
| Prompt loss | 0.51726 | 0.15446 | 0.20270 | 0.999987 |
| EOS masked as padding | 4.90538 | 0.06291 | 6.21310 | 0.995753 |
| Double label shift | 5.80103 | 5.72101 | 5.25259 | 0.053310 |
| Microbatch mean | 4.90510 | 0.06496 | 6.18896 | 0.999840 |

Prompt-loss training lowered prompt and template NLL while increasing answer NLL
and reducing check quality. Double shifting had the highest answer NLL and the
lowest EOS probability after a correct reference. Yet its generated answers all
terminated on EOS and all failed exact match. Generated termination frequency
and teacher-forced EOS probability therefore measured different behavior here.

The diagnostic contrasts add adapter evidence to the
[base-model logit audit](logit-audit-20261020.md). They have no fitted detection
thresholds or deployment likelihoods. B24's pretrained comparison is complete
for these four injections; the separate REST investigation is still pending.
