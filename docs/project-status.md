# Project status — 19 October 2026

Repository baseline: main `2988961`, including merged PRs through #98. PR #92
records the terminal `zero_sd` development stop; its CI passed. All six historical
development adapters scored 102/102. No fresh reference jobs, full persisted API
investigation or fresh final-test evaluation followed. The old protocol and its
prepared final test remain frozen. See [the execution record](fresh-api-execution-20261019.md).

The [development sensitivity follow-up](development-sensitivity-20261019.md)
audits the ceiling and rejects a separately frozen candidate: healthy mean
26.85%, damaged mean 39.81% on new histories. Its paid pilot stopped before
submission at the strict base gate. Additional provider spend is $0; the $5
budget retains its $0.2710464 historical charge estimate. Comparison evidence and mode
boundaries remain in the [fault review](fault-evidence-review.md).

A new format/content development design
separates clean format anchors from history content. Four arms (format, healthy,
cut and schema-valid wrong values) use three shuffled seeds each. At this
operational update, all 12 existing jobs remain running; the collector waits for
them, then evaluates adapters sequentially and applies the frozen gates. The
series reserves $1.1311488 within the inherited $5 ledger. Results and final-test
evidence remain pending; no additional jobs are needed to finish this series.

| Area | Evidence and current state | Remaining acceptance condition / owner |
|---|---|---|
| A2/A3 datasets | #82: JSON extraction and function calling, versioned splits and agent-reviewed references | Full task/model quality references and fresh evaluation; tracks A/B |
| A4 validation | Explicit-case API validates roles, nonblank text and user prompts before worker creation; tests cover all three tasks and four splits | Semantic duplicate detection remains outside this validation; track A |
| A7/A8/fault 1 | [Semantic record](fault-1-semantics.md), answer-cut probe and observed-row drop pair; length preflight is distinct | [Clean-prefix and alternative-malformation controls](answer-cut-specificity-controls-20261019.md) measured; independent likelihood calibration, causal specificity and multi-task evidence; tracks A/B |
| Measured fault 1 | #84: Kaggle and Token Factory target/control results with incomplete recovery; two new public experiment replays | Fresh complete API investigation and untouched final test; tracks A/B |
| B5 / A13 | Reference tooling and three datasets exist | Full three-task × three-model matrix; baseline/healthy/damaged/fixed metrics and budgeted plan; tracks A/B |
| B9 | #91: all 210 runs completed; local binary balanced accuracy 96.7%, historical Token Factory transfer 25% | Keep classifiers offline; independent provider-compatible transfer remains a B26 dependency |
| B12 | #85: development-fitted temperature worsens heldout NLL/Brier/ECE; T=1 remains default | New independent calibration evidence before deployment; track B |
| B17 | #85: SQLite reservations, REST/UI paid-step approval/resume; [recipe retrain reservations and persistent Tavily credits](paid-operation-recovery.md) | [SQLite worker ownership and duplicate-payment guards](sqlite-investigations.md) implemented; [provider training usage reconciled](provider-cost-reconciliation-20261019.md); new calls persist [provider request references](persisted-llm-usage.md), while invoices and per-call matching remain open; track B |
| B24 | Four fault injections, label audits and 60 synthetic CPU runs | Pretrained-LLM quality and complete mode-B investigation; track B |
| Public UI | Replay selection clears prior results and ignores stale loads; long loss histories use three axis labels and an expandable complete table; 36 saved reports | Deploy and verify the long-history layout fix; fresh trace still pending; this chat |
| Cause claims | README, verdict-v3 and versioned comparison metadata distinguish intervention benefit, unique cause and full recovery | New competing-cause evaluation; tracks A/B |
| O2 demo | [Walkthrough](demo-walkthrough.md) covers measured experiments and historical agent replay | Fresh full trace, public video and submission package; submission owner |

B9 analysis and publication are complete; do not repeat its 210 trainings or
reapply its earlier package. Its local sweep used no provider service. The B12
package separately records eight Super calls costing $0.0135807; that package is
outside this continuation's explicitly inherited $5 ledger. B9 confirmed MPS was
free before this follow-up; local evaluations run one process at a time.

## Next work, in order

1. Collect and evaluate the 12 already submitted format/content development jobs,
   apply the frozen admission/noise gates, then export and independently verify
   their results. Do not duplicate training or relax gates after seeing outcomes.
   Previous development stopping decisions remain terminal; their viewed checks
   are not final tests.
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
   six current-ledger costs match job receipts and 12 reservations remain pending.
   Obtain invoice evidence, resolve the $0.01 displayed-total difference and
   match inference calls when provider records permit. New calls now retain provider
   request/response IDs and their reservation IDs for that matching. SQLite worker ownership,
   recipe reservations and persistent Tavily credits are implemented.
5. Resolve supported scope for faults 2–5, embeddings/Nano/cleanlab, B24 pretrained
   confirmation and reproducible regression export with tracks A/B. B24 CPU
   synthetic evidence does not validate a pretrained model or a full mode-B trace.
6. The submission owner must confirm eligibility/registration, make the public
   video and submit the package. The repository cannot establish those facts.

Historical heldout cases have been viewed and reused; their metrics remain
exploratory evidence. The current development continuation has a periodic completion
check; its handoff record is in the ignored cohort artifacts. It must be disabled
after results are published.
