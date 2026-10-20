# Ticket-ledger admission stopped

All twelve LoRA adapters completed evaluation on 180 frozen development examples:
36 serialization anchors and 144 history questions, split evenly among copy,
restore and conditional tasks. Verification checked all 2,160 records against
their examples, recomputed metrics, checked adapter hashes and confirmed the
provider accepted LoRA mode.

The candidate failed the `healthy_minus_cut` admission gate. Healthy history
accuracy averaged 41.90%, versus 37.04% for cut answers: a 4.86-point difference,
below the required eight points. Every other frozen admission gate passed.
The decision is terminal for this design; it supplies no confirmation study or
fresh final-test result.

| Arm | Seed 2401 | Seed 2402 | Seed 2403 | Mean history exact |
| --- | ---: | ---: | ---: | ---: |
| Healthy | 44.44% | 43.06% | 38.19% | 41.90% |
| Cut | 34.72% | 36.81% | 39.58% | 37.04% |
| Format | 0.00% | 0.00% | 0.00% | 0.00% |
| Wrong value | 11.11% | 16.67% | 20.14% | 15.97% |

Every arm scored 100% on serialization anchors. Healthy mean accuracy was
40.28% for copy, 43.06% for restore and 42.36% for conditional histories.
Each arm has three distinct adapter digests. The seeds determine training-row
order and are also recorded in provider acceptance; this alone does not prove
independent provider initialization.

The [decision](results/ticket-ledger-outcome-20261020/decision.json),
[scores](results/ticket-ledger-outcome-20261020/scores) and
[accepted job records](results/ticket-ledger-outcome-20261020/accepted)
retain the evidence. The source decision's digest links the published outcome
to the local original; the published cost fields cover this study only.
The LoRA training token estimate is $1.5297504, with $1.83570048 charged to the
local budget including its allowance. The twelve cancelled Full SFT attempts
retain $2.3597568 in reservations pending provider billing.

The separate code-fault REST study began after these evaluations released MPS.
It has its own frozen protocol and does not depend on admitting this candidate.
