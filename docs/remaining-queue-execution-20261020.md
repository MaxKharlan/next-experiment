# Remaining queue execution, 20 October 2026

The user authorized autonomous continuation, including paid runs. Two separate
limits apply: $25 additional Token Factory spending and $30 AI Cloud spending.
Historical $5 reservations remain in their original ledger. Estimated costs and
unresolved holds are not invoices.

The additional Token Factory ledger is
`artifacts/remaining-queue-20261020/spending.sqlite3`; the separate cloud ledger is
`artifacts/remaining-queue-20261020/ai-cloud-spending.sqlite3`. At this checkpoint,
TF commitments total $4.261323 and cloud reservations total $24. Three qualification
attempts retain $4 each; a failed code-matrix bootstrap retains $6 and its corrected
execution reserves $6. Remaining balances are $20.738677 and $6 respectively. The
remaining cloud $6 is allocated to the data matrix after runtime verification.

| Work | Current state | Next evidence |
| --- | --- | --- |
| Ticket-ledger admission | Twelve correct LoRA jobs succeeded; serial MPS scoring is running | All twelve score files and frozen admission gates |
| Initial ledger execution | Twelve unintended Full SFT jobs cancelled; $2.3597568 reserve retained | Provider cancellation billing; invalid adapters excluded |
| A13/B5/B10 | Matrix inventory and budget envelope prepared | Admission, actual token budgets, selected recipes and noise study |
| Faults 2–5 | Injection/control helpers, 256-example data preparation and observed-label repair prepared | Prepared 36-cell CUDA cohort waits for the corrected code runtime |
| A10/A12 | Semantic overlap and distribution controls measured in #106 | Downstream quality and independent validation remain distinct |
| A19/A30/A31 | Automatic 200-example comparison complete in #105 | User replaced human review; failures and API errors retained |
| A29 | Twelve-reference teacher-forced diagnostic complete in #106 | Adapter diagnostics prepared in CUDA cohorts; live likelihoods remain unfitted |
| A15/A20 | Serverless creation and S3 retrieval work; explicit FP32/math SDPA passed the numerical gate | Kaggle generation notebook queued; full generation comparison remains open |
| B24 | Two supplemental MPS adapters measured; REST route implemented and tested | Corrected fifteen-cell CUDA bootstrap submitted; REST study waits for MPS and then runs once |
| A32 | FT-Fail published and verified; #107 merged, publication receipts in #108 | Complete for historical cases and automatic audit package |
| B21/B25/B26 | Published partial reproduction and negative transfer results retained | Missing original setups and independent provider-compatible transfer data |
| Billing | Usage still updated at 11:58 UTC; Payments has no documents | Later provider billing; no release of unresolved reserves |
| Submission | English description and requirements prepared; owner confirmed eligibility/registration/representation | Fresh trace, video, public YouTube link and final submission |

The original CUDA qualification failed at maximum NLL difference 0.003365 versus
its 0.0001 limit. A subsequent engineering diagnostic explicitly disabled TF32
and fused SDPA and passed the unchanged gates against MPS and CPU. The first
qualified code-matrix job then failed before training because inherited image
pip constraints conflicted with frozen NumPy. A new bootstrap isolates installer
configuration; hashes verify that its scientific payload is unchanged. All
failed attempts remain recorded with their budget holds.

The REST trace uses four new seeds per arm, a $1 request budget within the global
TF cap, saved SSE events and restart/idempotency verification. It starts only
after all twelve ledger score files are complete and their MPS lock is released.
An inconclusive diagnosis is a valid outcome; there are no automatic paid retries.

Custom weights beta is optional. The current path downloads adapters and evaluates
them locally, with cloud evaluation as a separately qualified backend. No support
message was sent. A PR records one package; it does not end the remaining queue.
