# Project status — 19 October 2026

Repository baseline: main `099aa16`, including merged PRs #82–#88. Main passed
741 tests (15 skipped), Ruff and formatting in this continuation. PR #88's
approved answer-drop retrain implementation is merged. The current follow-up
[prepares a new reviewed JSON cohort](fresh-answer-preparation.md) and fixes the
probe's failure above 10% malformed JSON. No fresh model quality is measured.
Comparison evidence and mode boundaries remain described in the
[fault-by-fault review](fault-evidence-review.md).

| Area | Evidence and current state | Remaining acceptance condition / owner |
|---|---|---|
| A2/A3 datasets | #82: JSON extraction and function calling, versioned splits and agent-reviewed references | Full task/model quality references and fresh evaluation; tracks A/B |
| A4 validation | Explicit-case API validates roles, nonblank text and user prompts before worker creation; tests cover all three tasks and four splits | Semantic duplicate detection remains outside this validation; track A |
| A7/A8/fault 1 | [Semantic record](fault-1-semantics.md), answer-cut probe and observed-row drop pair; length preflight is distinct | Calibrated likelihoods, competing-cause specificity, multi-task evidence; tracks A/B |
| Measured fault 1 | #84: Kaggle and Token Factory target/control results with incomplete recovery; two new public experiment replays | Fresh complete API investigation and untouched final test; tracks A/B |
| B5 / A13 | Reference tooling and three datasets exist | Full three-task × three-model matrix; baseline/healthy/damaged/fixed metrics and budgeted plan; tracks A/B |
| B9 | #85 publishes negative historical transfer; local 210-run sweep still active | Finish sweep, seed-grouped analysis and independent transfer; concurrent B9 chat |
| B12 | #85: development-fitted temperature worsens heldout NLL/Brier/ECE; T=1 remains default | New independent calibration evidence before deployment; track B |
| B17 | #85: SQLite reservations, REST/UI paid-step approval/resume | Answer-drop retrain now reserves its estimate; billing reconciliation, recipe retrain/Tavily scope and durable multiworker ownership remain; track B |
| B24 | Four fault injections, label audits and 60 synthetic CPU runs | Pretrained-LLM quality and complete mode-B investigation; track B |
| Public UI | Replay selection clears prior results and ignores stale loads; nested evidence is readable; 36 saved reports | #87 deployment verified against main; fresh trace still pending; this chat |
| Cause claims | README, verdict-v3 and versioned comparison metadata distinguish intervention benefit, unique cause and full recovery | New competing-cause evaluation; tracks A/B |
| O2 demo | [Walkthrough](demo-walkthrough.md) covers measured experiments and historical agent replay | Fresh full trace, public video and submission package; submission owner |

The concurrent B9 chat reported 144/210 completed runs during this preparation,
with no final analysis or fitted model yet. This is a dated checkpoint, not a live
counter. The local sweep has no provider charge; the B12 collection in #85 records
eight Super calls costing $0.0135807. The active training checkout is independent
from this follow-up.

## Next work, in order

1. Complete development noise calibration and freeze execution after the
   [fresh cohort preparation](fresh-answer-preparation.md). PR #88 is merged;
   the full persisted API trace and final-test quality remain unmeasured.
2. Complete B9 in its existing chat. Add final results on the current main; do not
   reapply the preliminary 37-file package already merged in #85.
3. Use the [fresh evaluation protocol](fresh-evaluation-protocol.md) for the
   answer-cut API trace. Reviewed data hashes, tokenizer counts and a $5-cap plan
   now exist; pin runtime revisions, verify provider pricing and select seeds
   from development noise before submission. Avoid a second
   model load or MPS evaluation while B9 uses the shared 16 GB machine.
4. Calibrate signatures/noise on development data. Keep T=1 and the experimental
   classifier out of default live diagnosis until independent evidence supports them.
5. Decide the supported scope for faults 2–5, embeddings/Nano/cleanlab, B24
   pretrained confirmation, and reproducible regression export.
6. Confirm submission ownership/eligibility, create the final public video and
   complete submission. Repository changes cannot establish registration status.

No new paid experiments were started during this follow-up. Historic heldout cases
have been viewed and reused; their metrics are not fresh generalization evidence.
