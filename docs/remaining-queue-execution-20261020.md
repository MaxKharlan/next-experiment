# Remaining queue execution, 20 October 2026

The user authorized autonomous continuation, including paid runs. Two separate
limits apply: $25 additional Token Factory spending and $30 AI Cloud spending.
Historical $5 reservations remain in their original ledger. Estimated costs and
unresolved holds are not invoices.

The additional Token Factory ledger is
`artifacts/remaining-queue-20261020/spending.sqlite3`; the separate cloud ledger is
`artifacts/remaining-queue-20261020/ai-cloud-spending.sqlite3`. At this checkpoint,
Cloud reservations now total $30: three qualification attempts at $4 and three
matrix attempts at $6. The last matrix attempt combines the code and data cohorts
in one job. No further cloud launch is budgeted. The Qwen clean-reference grid
adds up to $5.917018 of TF reservations to the previous $4.261323 commitments;
each job reserves before dispatch. This leaves room for the separate $1 REST trace.
See the timestamped progress JSON for current committed amounts.

| Work | Current state | Next evidence |
| --- | --- | --- |
| Ticket-ledger evaluation | [All twelve adapters and 2,160 records verified](ticket-ledger-outcome-20261020.md) | Evaluation complete |
| Initial ledger execution | Twelve unintended Full SFT jobs cancelled; $2.3597568 reserve retained | Provider cancellation billing; invalid adapters excluded |
| A13/B5/B10 | Actual token budgets prepared; 24 healthy-reference Qwen grid jobs dispatched serially in batches of three | Validation selection, selected-check quality; Llama inference route and noise study |
| Faults 2–5 | [29/36 CUDA cells verified](data-fault-quality-20261020.md); frozen data timeout reached; all seven healthy-adapter validation evaluations completed for three seeds | Seven seed-6103 cells remain missing; no automatic retry or additional Cloud launch |
| A10/A12 | Semantic overlap and distribution controls measured in #106 | Downstream quality and independent validation remain distinct |
| A19/A30/A31 | Automatic 200-example comparison complete in #105 | User replaced human review; failures and API errors retained |
| A29 | Base diagnostic in #106 and sixteen-reference measurements for fifteen code and 29 data CUDA adapters complete | Independent calibration remains open; live likelihoods remain unfitted |
| A15/A20 | Serverless creation and S3 retrieval work; [all 64 base-model records agree across MPS, Cloud L40S and Kaggle T4](kaggle-runtime-comparison-20261020.md); all 46 Kaggle preflight checks passed | Same-checkpoint development comparison complete; training/backend equivalence is outside this result |
| B24 | Two supplemental MPS adapters and [all fifteen CUDA code cells](pretrained-code-quality-20261020.md) verified; fresh REST study `inv-1f6c0581` started after ledger scoring | Saved report, SSE and restart/idempotency verification pending |
| A32 | FT-Fail published and verified; #107 merged, publication receipts in #108 | Complete for historical cases and automatic audit package |
| B21/B25/B26 | Published partial reproduction and negative transfer results retained | Missing original setups and independent provider-compatible transfer data |
| Billing | October billing export contains Object Storage usage through 12:00 UTC; GPU charges are absent | Later provider billing; no release of unresolved reserves |
| Submission | English description prepared; owner confirmed eligibility/registration/representation; silent captioned video and text PDF saved in Downloads; updated feedback stays local for owner submission | Fresh trace, public YouTube link and final submission |

The original CUDA qualification failed at maximum NLL difference 0.003365 versus
its 0.0001 limit. A subsequent engineering diagnostic explicitly disabled TF32
and fused SDPA and passed the unchanged gates against MPS and CPU. The first
qualified code-matrix job then failed before training because inherited image
pip constraints conflicted with frozen NumPy. A new bootstrap isolates installer
configuration; hashes verify that its scientific payload is unchanged. All
failed attempts remain recorded with their budget holds. A second code attempt
completed base predictions but failed while copying filesystem metadata to the
bucket mount. The final combined job reuses those verified predictions, exports
only bytes. It completed all fifteen code cells and 29 data cells before the
data subprocess's frozen 3,600-second limit. All 535 collected files were
verified. The provider's terminal job status is `FAILED`; the data worker's
last `running` snapshot is preserved as historical state.

The REST trace uses four new seeds per arm, a $1 request budget within the global
TF cap, saved SSE events and restart/idempotency verification. It starts only
after all twelve ledger score files are complete and their MPS lock is released.
An inconclusive diagnosis is a valid outcome; there are no automatic paid retries.

Custom weights beta is optional. The current path downloads adapters and evaluates
them locally, with cloud evaluation as a separately qualified backend. This path
is available for Qwen. Llama 1B has provider training access, but its LoRA export
lacks base weights and the inference catalog has no matching endpoint. The user
selected Token Factory; its tokenizer was retrieved, while the Llama quality grid
remains unsubmitted until an evaluation route exists. No support
message was sent. A PR records one package; it does not end the remaining queue.
