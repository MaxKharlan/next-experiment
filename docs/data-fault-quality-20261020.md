# Partial data-fault quality results

The Cloud data study stopped at its frozen 3,600-second subprocess limit after
29 of 36 planned cells. The parent job ended as `FAILED`. All 535 collected
artifacts passed hash verification, and the saved metrics were recomputed from
predictions. The code study in the same job had already completed all 15 cells.
The [terminal record](results/cloud-matrices-export-20261020/terminal-summary.json)
lists the seven missing data cells; every missing cell uses seed 6103. The raw
data worker's last snapshot says `running`, while the parent result records the
timeout. Both records are retained.

No retry follows this terminal result. The protocol permits one attempt, and the
$30 Cloud limit remains fully reserved pending billing.

## Quality on the fixed check set

All cells use Qwen3-0.6B, three epochs, learning rate 2e-4, rank/alpha 16,
batch size two and gradient accumulation four. The check set contains 128
BANKING77 examples across eight classes. Each completed cell ended every
generation on EOS without reaching the token limit.

| Training arm | Completed seeds | Mean exact match |
| --- | ---: | ---: |
| Healthy | 3 | 91.41% |
| Foreign template, 25% | 3 | 90.89% |
| Foreign template, 50% | 3 | 90.36% |
| Foreign-template control | 3 | 91.67% |
| Contradictory labels, 25% | 3 | 82.55% |
| Contradictory labels, 50% | 2 | 67.58% |
| Label-removal control | 2 | 66.41% |
| Observed-label repair | 2 | 84.77% |
| Mild distribution shift | 2 | 88.67% |
| Strong distribution shift | 2 | 82.81% |
| Shift control | 2 | 77.73% |
| Shift repair | 2 | 82.03% |

The [analysis](results/cloud-matrices-export-20261020/analysis.json) retains
per-seed scores, sample standard deviations and comparisons restricted to
matching completed seeds. Means with two seeds must not be subtracted from
three-seed means to calculate a paired effect.

Foreign-template corruption produced small, inconsistent changes: healthy minus
25% corruption was −1.56, 0 and +3.13 percentage points; healthy minus 50% was
−3.91, +3.13 and +3.91 points. This setting did not produce a consistent harmful
effect. Exact reconstruction of the declared wrapper yields the healthy
training data; its mean advantage over the shuffled corrupted control was
−0.26 points.

Contradictory labels reduced quality at both measured strengths. On the two
completed matched seeds, observed-label repair beat random equal-count removal
by 21.09 and 15.63 points, a mean of 18.36 points. Repair remained 2.34 and
9.38 points below healthy. Both removal arms retained 141 of 256 examples:
repair used 46,992 input and 3,213 supervised tokens over training, versus
46,632 and 3,204 for the control. Equal retained example counts still produced
different training token counts.

Distribution repair beat its shuffled control by 3.13 and 5.47 points, but
scored 1.56 and 0 points below the unrepaired strong-shift arm. Resampling
therefore did not improve the observed damaged baseline. All three arms used
256 examples; repair used 84,168 input and 5,856 supervised tokens, versus
80,985 and 4,533 for the strong-shift arm and its control.

## Validation overlap

The three healthy adapters also completed all seven validation evaluations.
Training was unchanged. Overlap was injected into validation, then detected
prompt overlaps or an equal number of random rows were removed.

| Validation set | Examples | Mean exact match |
| --- | ---: | ---: |
| Clean | 64 | 91.67% |
| 25% overlap | 64 | 92.71% |
| 25% overlap, deduplicated | 48 | 90.28% |
| 25% overlap, random removal | 48 | 94.44% |
| 50% overlap | 64 | 96.35% |
| 50% overlap, deduplicated | 32 | 92.71% |
| 50% overlap, random removal | 32 | 100.00% |

Contaminated validation exceeded clean validation by a mean of 1.04 points at
25% overlap and 4.69 points at 50%. Deduplication reduced the contaminated
score, while random removal retained more of the optimism. The remaining sets
contain different examples and sizes; these results measure validation bias
in this construction, with no new training-quality effect.

All 29 completed adapters include sixteen-reference teacher-forced NLL and EOS
diagnostics in the [measurements](results/cloud-matrices-export-20261020/measurements/data).
These descriptive development results add no fitted live fault likelihoods.
Seven planned seed-6103 measurements and broader strength curves remain
unmeasured. The timeout, two-seed comparisons and fixed check population limit
the claims supported by this study.
