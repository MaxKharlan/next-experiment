# Teacher-forced logit diagnostics

A29 now has a level-1 probe and a measured Qwen3-0.6B development example.
The probe separates prompt-body, answer-body and special-template-token loss,
uses the causal shift `logits[i-1] → token[i]`, and measures EOS probability
immediately after the supplied reference answer. It rejects truncation, template
prefix disagreement, overlapping token regions and nonfinite logits.

The [frozen selection](results/logit-audit-20261020/selection.json) uses three
serialization anchors and the first changed history from each ticket-ledger
family, with the same three history prompts under healthy, cut and wrong-value
references. All twelve examples come from training data. This is a mechanism
illustration, not an independent diagnostic validation set.

| Reference group | Answer NLL per token | Mean EOS probability after reference |
| --- | --- | --- |
| Healthy serialization | 0.1002 | 0.999993 |
| Healthy history | 0.6827 | 0.999987 |
| Cut history | 1.3006 | 6.57e-12 |
| Wrong-value history | 0.8660 | 0.999984 |

A low EOS probability distinguishes these incomplete strings, while full strings
with wrong values retain a high EOS probability. It cannot establish semantic
correctness. Prompt NLL is 1.6605 across the pooled token set. Special-template
perplexity is 4015.1; it includes special delimiters at multiple conversation
boundaries, not only the final EOS, and is not ordinary prompt perplexity.

The [raw rows, runtime and hashes](results/logit-audit-20261020/base/) preserve
the measurement. `TeacherForcedLogits` is an opt-in `Probe` requiring explicit
local data, model snapshot, context length and optional adapter paths. Its
single outcome, `measured`, has a neutral signature row of 1.0. It is excluded
from automatic probe selection until independent fault-conditioned calibration
exists. No likelihoods were invented from these twelve authored examples.

The implementation and local measurement portions of A29 are complete.
Fault-1/fault-2 discrimination and transfer-calibrated signatures remain separate
research acceptance conditions. New LoRA measurements must name their checkpoint
and use the frozen data; completed model/reference pairs are not rerun to improve
these numbers.
