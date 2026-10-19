# Project status — 19 October 2026

Repository baseline: main `85aeebe`, including merged PRs #82–#85. CI and Pages
succeeded on that commit. At this review there were no open PRs or GitHub Issues.
This follow-up adds validation/replay fixes, measured answer-cut replays and the
fault-semantics clarification. Implementation coverage and experimental coverage
are separate below.

| Area | Evidence and current state | Remaining acceptance condition / owner |
|---|---|---|
| A2/A3 datasets | #82: JSON extraction and function calling, versioned splits and agent-reviewed references | Full task/model quality references and fresh evaluation; tracks A/B |
| A4 validation | Explicit-case API validates roles, nonblank text and user prompts before worker creation; tests cover all three tasks and four splits | Semantic duplicate detection remains outside this validation; track A |
| A7/A8/fault 1 | [Semantic record](fault-1-semantics.md), answer-cut probe and observed-row drop pair; length preflight is distinct | Calibrated likelihoods, competing-cause specificity, multi-task evidence; tracks A/B |
| Measured fault 1 | #84: Kaggle and Token Factory target/control results with incomplete recovery; two new public experiment replays | Fresh complete API investigation and untouched final test; tracks A/B |
| B5 / A13 | Reference tooling and three datasets exist | Full three-task × three-model matrix; baseline/healthy/damaged/fixed metrics and budgeted plan; tracks A/B |
| B9 | #85 publishes negative historical transfer; local 210-run sweep still active | Finish sweep, seed-grouped analysis and independent transfer; concurrent B9 chat |
| B12 | #85: development-fitted temperature worsens heldout NLL/Brier/ECE; T=1 remains default | New independent calibration evidence before deployment; track B |
| B17 | #85: SQLite reservations, REST/UI paid-step approval/resume | Billing reconciliation, retrain/Tavily scope and durable multiworker ownership; track B |
| B24 | Four fault injections, label audits and 60 synthetic CPU runs | Pretrained-LLM quality and complete mode-B investigation; track B |
| Public UI | Replay selection clears prior results and ignores stale loads; nested evidence is readable; 36 saved reports | Publish this follow-up and verify deployment; this chat |
| Cause claims | README and new prompts distinguish intervention benefit, unique cause and full recovery; legacy reports stay intact | New competing-cause evaluation; tracks A/B |
| O2 demo | [Walkthrough](demo-walkthrough.md) covers measured experiments and historical agent replay | Fresh full trace, public video and submission package; submission owner |

The concurrent B9 chat reported 94/210 completed runs at 10:48 Moscow time,
with no final analysis or fitted model yet. This is a dated checkpoint, not a live
counter. The local sweep has no provider charge; the B12 collection in #85 records
eight Super calls costing $0.0135807. The active training checkout is independent
from this follow-up.

## Next work, in order

1. Publish and verify the validated follow-up, including the two measured replays.
2. Complete B9 in its existing chat. Add final results on the current main; do not
   reapply the preliminary 37-file package already merged in #85.
3. Use the [fresh evaluation protocol](fresh-evaluation-protocol.md) to prepare
   new task/model references and an answer-cut API trace. Finalize data hashes,
   seed plan, stopping rule and cost cap before provider submission.
4. Calibrate signatures/noise on development data. Keep T=1 and the experimental
   classifier out of default live diagnosis until independent evidence supports them.
5. Decide the supported scope for faults 2–5, embeddings/Nano/cleanlab, B24
   pretrained confirmation, and reproducible regression export.
6. Confirm submission ownership/eligibility, create the final public video and
   complete submission. Repository changes cannot establish registration status.

No new paid experiments were started during this follow-up. Historic heldout cases
have been viewed and reused; their metrics are not fresh generalization evidence.
