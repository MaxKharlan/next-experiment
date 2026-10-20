# Remaining scope reconciled to the original plan

This reconciliation uses the project document dated 16 October 2026, pages
30–34 (A1–A32 and B1–B26), and the verified terminal data, ledger, REST and Qwen results, with execution tracked in
[the execution queue](remaining-queue-execution-20261020.md). The document's
checkboxes are planning history. Implementations, measurements and publication
are separate acceptance conditions.

## Data faults and semantic checks

| Requirement | Supported now | Remaining acceptance condition |
| --- | --- | --- |
| A4 message validation | Roles, nonempty text, user prompts, structured contracts | Semantic duplicate detection belongs to A10; it does not block A4's stated format contract |
| Fault 2 / A5,A7,A9 | Declared-wrapper injection, reconstruction and matched control measured on three seeds; no consistent harm in this setting | Other model/task coverage and broader strength curves |
| Fault 3 / A5,A7,A11 | Semantic audit and observed-label repair/control measured; two completed repair seeds | Missing seed 6103, broader ambiguity and task/model coverage |
| Fault 4 / A5,A7,A10 | Lexical/semantic controls and three-seed validation-optimism measurements with deduplication/random-removal controls | Independent semantic validation and broader task/model coverage |
| Fault 5 / A5,A7,A12 | Declared target distribution and embedding checks; two seeds of shift/repair/control quality, with no improvement over the damaged baseline | Missing seed 6103 and broader strength/task/model coverage |
| A18 sensitivity | Answer-cut evidence and [partial two-strength data-fault measurements](data-fault-quality-20261020.md) | Missing cells and justified broader strength curves; a planted setting alone is not a harmful fault |
| A19 Nano audit | [200-example automatic comparison](semantic-audit-20261020.md); manual review replaced by user instruction | Complete under revised acceptance; preserve Nano failure and Super HTTP errors |
| A29 logit probes | Teacher-forced NLL/EOS on base references plus fifteen code and 29 data adapters; neutral live likelihoods | Independent calibration |
| A30 embeddings | [Local Qwen, NVIDIA and provider Qwen measured](semantic-audit-20261020.md), licenses and outcome agreement recorded | Complete for the frozen development cases |
| A31 cleanlab | [Five-fold logistic regression + cleanlab comparison](semantic-audit-20261020.md), including Nano union | Complete on the revised automatic reference |

A19/A30/A31 now have a [measured automatic audit](semantic-audit-20261020.md).
The user replaced human annotation with automated checking; the original
requirement and acceptance amendment remain visible. These research results
do not introduce calibrated live fault likelihoods.
The lexical baseline now declares its representation and threshold and publishes
paraphrase misses and false positives. It remains a candidate review tool; no
semantic-detection accuracy or deployment likelihood has been established.

## Research and transfer

The [3×3 inventory and budget](reference-matrix-plan-20261020.md) separates the
new Qwen clean-reference measurements from missing Llama, damaged/fixed/control
and B10 noise coverage. The
[ticket-ledger bundle](ticket-ledger-development-20261020.md) prepares a new
candidate; [all twelve adapters have now been evaluated](ticket-ledger-outcome-20261020.md).

Answer-cut's 35 synthetic controls already show clean-prefix false alarms and
alternative malformed-JSON causes. Independent likelihood calibration requires
an independently sampled, labeled development population with clean cases,
actual cuts, internal syntax errors, wrong values and recipe-only failures.
Sampling weights and train/calibration/evaluation boundaries must be frozen.
Do not fit deployment likelihoods from the balanced constructed controls or
from the new ledger's deterministic parser errors. Multi-task model-quality
and competing-cause evidence remains open.

B25 has a weight probe and documented failed scale transfer; a smaller rank is
not necessarily a harmful fault. B26's 25% historical provider transfer rejects
live deployment. New provider-compatible independent data are needed for both;
relabeling or resplitting previously inspected runs cannot create independence.
Keep both classifiers offline and T=1. B9's 210-run analysis and B12's negative
calibration decision are complete and must not be repeated.

B21's five public reports are collected. TRL #696 lacks the full original data
and pinned environment. The [isolated historical #38837 experiment](transformers-38837-reproduction-20261020.md)
provides partial CPU mechanism evidence: four of six case/rate comparisons agree,
but accumulation-3 controls fail. Its prespecified status remains `stopped`.
Original-case and downstream-quality reproduction are still missing.
The remaining reports lack a confirmed cause or reproducible original setup.
Record synthetic mechanism reproductions separately from original-case
reproductions and downstream quality evidence.

B24's four injections, audits and 60 synthetic runs are implemented and measured.
[Fifteen pretrained CUDA comparisons](pretrained-code-quality-20261020.md) and
the [REST mode-B investigation](rest-code-trace-20261020.md) are complete.
The REST diagnosis is inconclusive; its eight adapter scores, six SSE events,
restart recovery and idempotent POST are verified. Arbitrary user-code repair is an unvalidated extension; it is not the
original B24 acceptance condition, which names four injections and code checks.

## Product, evidence and external dependencies

The original A27 requests Markdown/PDF reports with artifact/source links;
A24 requests a benchmark runner, budgets and uncertainty. They do not create an
extra universal regression-export product requirement. Portable deterministic
exports exist for the published format/content series and new ledger preparation.
A broad cross-suite export remains useful implementation work if required by a
specific regression workflow; it should not be silently added to the original
acceptance criteria. A32 is published as [MakHH/FT-Fail](https://huggingface.co/datasets/MakHH/FT-Fail),
with all 35 package files verified from the pinned Hub revision. B26 classifier
publication remains a distinct deliverable with unmet transfer validation.

The long-history fix was verified on the published desktop demo on 20 October;
see [the walkthrough](demo-walkthrough.md). A silent video and its text PDF are saved in Downloads.
Fresh REST evidence is published; video publication and submission remain open.
Cloud Sandbox activation remains an external
dependency; the local Docker path is available.

Billing needs newer provider Usage/invoice records, cancellation charge evidence
and inference per-call matching. The sole historical ledger stays at
`artifacts/format-content-development-20261019-v1/spending.sqlite3` in the
execution checkout. No preparation or plan here moves its holds, copies its
balance or treats an estimate as an invoice.

The owner confirmed eligibility, registration and authority to submit on
20 October. Final submission has not been performed; the remaining materials
are tracked in [submission readiness](submission/readiness.md).

## Remaining work after the current measurements

The code cohort, terminal partial data cohort, ledger evaluation, fresh REST trace,
runtime comparison and automatic semantic audit are published. Qwen reference
results are tracked in the [quality report](qwen-reference-quality-20261020.md).
Their completion does not fill the seven missing Cloud cells or the original
matrix's damaged/fixed/control and noise studies.

Further research needs a separately frozen protocol and budget: broader fault
strengths, missing model/task comparisons, size/seed noise measurements, independent
calibration and transfer. Missing original reproduction setups also remain open.
Do not repeat completed trainings or spend unresolved holds on those extensions.

External dependencies are a matching Llama evaluation route, independent
provider-compatible transfer data and later billing evidence. The owner still
needs to publish the video, send the local feedback, select the submission track
and complete Devpost submission. Custom weights beta is optional: downloaded
adapters evaluated locally or on Cloud supply the current Qwen path.
