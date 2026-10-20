# Ticket-ledger development preparation, 20 October 2026

The current v2 bundle defines the data, endpoints and evaluation settings.
The completed twelve-adapter study is recorded in the
[evaluation report](ticket-ledger-outcome-20261020.md). Preparing this bundle
performs no training, inference or spending reservation.

## Task and controls

Each history contains four ticket records and 18 numbered events. The output is
the complete requested ticket, including callback and due date. Three equally
represented families require conditional updates, restoration of an earlier
field value, or copying another ticket's value at the time of the event. Later
changes to a source ticket must not retroactively change a copied value.

| Split | Serialization anchors | Histories | Histories per family |
| --- | ---: | ---: | ---: |
| Train | 72 | 288 | 96 |
| Validation | 24 | 72 | 24 |
| Check | 36 | 144 | 48 |

The data contain 504 distinct normalized histories and 2,148 distinct visible
ticket IDs across splits. Grammar is shared. Direct anchors can repeat normalized
field combinations; they test serialization, not semantic generalization.
No final-test split is created, read or used for data design.

Four training arms retain the earlier scientific distinction: format-only,
healthy, cut and wrong-value. Cut and wrong affect exactly the same 144 history
rows, balanced by family; anchors remain clean. Cut removes the last half of
answer characters. Wrong cycles product and priority while preserving valid JSON.
All arms use common clean validation and fixed epochs. Format-only inputs are
shorter and do not provide a compute-matched causal control.

## Prospective measurements

The primary endpoint is strict full-object exact match over all check histories.
There is no candidate list supplying correct values for the other fields.
Schema validity, field accuracy, family results and generation-limit hits are
secondary. Missing, invalid and limit-hit answers remain in the denominator.

The proposed first model is Qwen3-0.6B: two epochs, learning rate 1e-4,
batch 8, LoRA rank/alpha 16, dropout .05, context 8192, packing off.
Order seeds are 2401, 2402 and 2403. Generation is greedy, thinking off,
with a fixed 192-token allowance. Every arm/seed is planned prospectively.

All admission gates must pass together on arithmetic means across the three seeds:

- Format and healthy anchor exact at least 90%.
- Healthy history exact between 40% and 95%, inclusive.
- Healthy minus format history exact at least 15 percentage points.
- Healthy minus wrong history exact at least 10 percentage points.
- Healthy mean within each family at least 25%.
- Three distinct weight digests per arm.

Any failed gate terminates this design. Missing runs are incomplete. Check results
cannot select recipes, extra seeds or replacement endpoints. No power estimate or
confirmation sample size is selected here. Passing admission would permit a
separate noise/power plan; it would not authorize a final-test run or paid work.
The new endpoint is explicit in this prospective study and does not reinterpret
the failed choice endpoint of the previous study.

## Offline review

The generator computes labels through forward state snapshots. A separate
text-only parser resolves each output field backwards through dependencies.
All 504 history labels agree. Hand-written tests cover false conditions,
copy-before-source-change, historical restoration and entity renaming. These
are two implementations by one author, with no independent human annotation.

Three deliberately incorrect parsers establish that the relevant operations
matter. On their respective 48-row check families, unconditional assignment gets
7 exact, ignoring restoration gets 2, and reading a copied source after its next
write gets 8. These are software controls, not measurements of model difficulty,
noise, causal specificity or deployment likelihood.

The [frozen bundle](results/ticket-ledger-development-20261020-v2/protocol.json)
contains six datasets, the corruption indices, audit, source/data hashes and the
protocol checksum. Verification regenerates every byte, including the review,
and rejects changed gates, data, unexpected files and source drift.

```bash
python -m scripts.prepare_ticket_ledger prepare --out /new/output/directory
python -m scripts.prepare_ticket_ledger verify \
  --out docs/results/ticket-ledger-development-20261020-v2
```

Use the existing project Python environment. `prepare` refuses an existing
output directory. Neither command has a provider client or a submission action.

The separate [token preflight](results/ticket-ledger-preflight-20261020/token-counts.json)
uses the existing local tokenizer only and records its revision and hashes.
The maximum prompt plus generation allowance is 873 tokens. At the historical
$0.40/M training rate, the 12-slot rendered-token estimate is $1.5332064; the
maximum-row reservation estimate is $1.966464. Neither includes inference, tax,
retries, contingency or the old pending holds. Current prices remain unverified.
Reproduce outside the frozen bundle with:

```bash
HF_HUB_OFFLINE=1 python -m scripts.ledger_token_preflight \
  --root docs/results/ticket-ledger-development-20261020-v2 \
  --tokenizer /local/Qwen3-0.6B/snapshot \
  --out /new/token-counts.json
```

Before model measurement, freeze the execution code, model revision and runtime.
Before paid submission, verify current prices, obtain a new explicit budget,
create per-attempt reservations and retain unresolved historical holds.
The old spending ledger is untouched; the preparation creates no successor ledger.
T=1 and the offline-only classifier decision remain in force.
