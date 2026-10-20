# Project status — 20 October 2026

Current continuation: [execution queue](remaining-queue-execution-20261020.md).
PRs #105–#108 publish the automatic semantic audit, neutral logit diagnostics,
semantic controls and verified FT-Fail Hub release. The owner authorized separate
$25 additional Token Factory and $30 AI Cloud limits and confirmed submission
eligibility, registration and representation. Prospective model scoring and
REST/cloud studies are running; their uncompleted outcomes are not benchmark
results. [Submission materials](submission/readiness.md) track the remaining
video and final submission.

The historical baseline below is main `1350564`, through #103. PR #92
records the terminal `zero_sd` development stop; its CI passed. All six historical
development adapters scored 102/102. No fresh reference jobs, full persisted API
investigation or fresh final-test evaluation followed. The old protocol and its
prepared final test remain frozen. See [the execution record](fresh-api-execution-20261019.md).

The [development sensitivity follow-up](development-sensitivity-20261019.md)
audits the ceiling and rejects a separately frozen candidate: healthy mean
26.85%, damaged mean 39.81% on new histories. Its paid pilot stopped before
submission at the strict base gate. That pilot added $0 provider spend; the historical $5
budget retained its $0.2710464 historical charge estimate at that stop. Comparison evidence and mode
boundaries remain in the [fault review](fault-evidence-review.md).

The [format/content development series](format-content-development-20261019.md)
completed twelve replacements after twelve original jobs were cancelled. All 13
models (including base), 936 answers and 624 choice decisions are exported and verified.
Mean history exact is 12.50% format, 100% healthy, 94.44% cut and 76.39% wrong. Three
frozen gates fail: healthy choice ceiling, insufficient healthy/wrong choice gap,
and insufficient healthy/cut strict gap. This is a terminal development stop;
no confirmation seeds or fresh final test are selected. New training token estimates
are $0.8469312, with $1.1311488 cancelled-job holds retained pending billing.

| Area | Evidence and current state | Remaining acceptance condition / owner |
|---|---|---|
| A2/A3 datasets | #82: JSON extraction and function calling, versioned splits and agent-reviewed references | Full task/model quality references and fresh evaluation; tracks A/B |
| A4 validation | Explicit-case API validates roles, nonblank text and user prompts before worker creation; tests cover all three tasks and four splits | [Lexical overlap audit](lexical-overlap-audit-20261020.md) is separate; semantic validation and fault-4 quality evidence remain open; track A |
| A7/A8/fault 1 | [Semantic record](fault-1-semantics.md), answer-cut probe and observed-row drop pair; length preflight is distinct | [Clean-prefix and alternative-malformation controls](answer-cut-specificity-controls-20261019.md) measured; independent likelihood calibration, causal specificity and multi-task evidence; tracks A/B |
| Measured fault 1 | #84: Kaggle and Token Factory target/control results with incomplete recovery; two new public experiment replays | Fresh complete API investigation and untouched final test; tracks A/B |
| B5 / A13 | Reference tooling and three datasets exist | Full three-task × three-model matrix; baseline/healthy/damaged/fixed metrics and budgeted plan; tracks A/B |
| B9 | #91: all 210 runs completed; local binary balanced accuracy 96.7%, historical Token Factory transfer 25% | Keep classifiers offline; independent provider-compatible transfer remains a B26 dependency |
| B12 | #85: development-fitted temperature worsens heldout NLL/Brier/ECE; T=1 remains default | New independent calibration evidence before deployment; track B |
| B17 | #85: SQLite reservations, REST/UI paid-step approval/resume; [recipe retrain reservations and persistent Tavily credits](paid-operation-recovery.md) | [SQLite worker ownership and duplicate-payment guards](sqlite-investigations.md) implemented; [provider training usage reconciled](provider-cost-reconciliation-20261019.md); new calls persist [provider request references](persisted-llm-usage.md), while invoices and per-call matching remain open; track B |
| B21 | [Historical Trainer CPU experiment](transformers-38837-reproduction-20261020.md): 4/6 comparisons agree; accumulation-3 controls fail; frozen status `stopped` | Full original-case reproduction and downstream quality; track B |
| B24 | Four fault injections, label audits and 60 synthetic CPU runs | Pretrained-LLM quality and complete mode-B investigation; track B |
| Public UI | Replay selection clears prior results and ignores stale loads; long loss histories use three axis labels and an expandable complete table; 36 saved reports | Published desktop Safari long-history and replay-switch checks passed on 20 October; fresh trace/video still pending; this chat |
| Cause claims | README, verdict-v3 and versioned comparison metadata distinguish intervention benefit, unique cause and full recovery | New competing-cause evaluation; tracks A/B |
| O2 demo | [Walkthrough](demo-walkthrough.md) covers measured experiments and historical agent replay | Fresh full trace, public video and submission package; submission owner |

B9 analysis and publication are complete; do not repeat its 210 trainings or
reapply its earlier package. Its local sweep used no provider service. The B12
package separately records eight Super calls costing $0.0135807; that package is
outside this continuation's explicitly inherited $5 ledger. B9 confirmed MPS was
free before this follow-up; local evaluations run one process at a time.

The [ticket-ledger preparation](ticket-ledger-development-20261020.md) now freezes
504 fresh histories, complete-object endpoints and new development gates.
Two offline oracles agree; tokenizer context checks pass. Model admission remains
`not_measured`; no new paid series is authorized.

The [3×3 matrix plan](reference-matrix-plan-20261020.md) records all nine cells,
missing comparable metrics and a separate prospective budget scenario.
The [scope reconciliation](remaining-scope-20261020.md) distinguishes original
A19/A30/A31 requirements from optional extensions and lists remaining offline work.

The [A10 lexical audit](lexical-overlap-audit-20261020.md) now emits train/validation
candidates with source hashes and line numbers. Its 22 authored controls expose
four false positives and four missed paraphrases; it has no live fault signature.

## Next work, in order

1. The format/content series is complete and rejected for confirmation. Use its
   exported diagnostics for a separately preregistered development data design
   that tests harder, fresh histories before any new paid series. Do not adapt
   this cohort's thresholds or repeat seeds. Its viewed checks are not final tests.
2. Only after a suitable new development design passes its noise/power gates,
   freeze a new confirmation protocol: fresh reference/control comparisons,
   persisted API trace with paid approvals and restart recovery, then final test
   after the intervention decision. One task/model study does not close the 3×3 matrix.
3. Calibrate signatures and competing-cause specificity on development evidence.
   [35 synthetic probe controls](answer-cut-specificity-controls-20261019.md) now
   demonstrate clean-prefix false alarms and alternative JSON causes; these
   constructed counts do not estimate deployment likelihoods.
   Keep T=1 and both experimental B9 classifiers out of live diagnosis.
4. Complete the remaining [billing reconciliation](provider-cost-reconciliation-20261019.md):
   224 completed training jobs match aggregate Usage ($8.0379272 before tax);
   the new series adds $0.8469312 in token estimates, while 12 cancelled originals
   retain $1.1311488 in holds. Their newer aggregate Usage match remains open.
   Obtain invoice evidence, resolve the $0.01 displayed-total difference and
   match inference calls when provider records permit. New calls now retain provider
   request/response IDs and their reservation IDs for that matching. SQLite worker ownership,
   recipe reservations and persistent Tavily credits are implemented.
5. Implement the remaining faults 2–5 probes and controls, A29 logit probes, and
   embeddings/Nano/cleanlab research from the reconciled scope. Continue B24
   pretrained confirmation under a separate frozen protocol with tracks A/B. B24 CPU
   synthetic evidence does not validate a pretrained model or a full mode-B trace.
6. The submission owner must confirm eligibility/registration, make the public
   video and submit the package. The repository cannot establish those facts.

Historical heldout cases have been viewed and reused; their metrics remain
exploratory evidence. Local MPS evaluation is complete. The periodic completion
check was disabled after publication in PR #100.
