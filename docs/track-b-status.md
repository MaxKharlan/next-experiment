# Track B reconciliation

Compared the original project plan with `main` through PR #92
(`2dff394`) and the repository implementation on October 19, 2026.
Checkboxes in the original plan do not establish that every acceptance condition
has measured evidence.

The current main also includes #92 (`2dff394`): the fresh answer-cut protocol
stopped at `zero_sd` before reference jobs or a persisted API trace. See the
[separate development sensitivity audit](development-sensitivity-20261019.md),
which also completed with a rejected candidate and $0 additional provider spend.
[35 probe controls](answer-cut-specificity-controls-20261019.md) demonstrate
clean-prefix false alarms and alternative causes of broken JSON; they do not
calibrate deployment likelihoods.

| Plan items | Existing implementation and remaining work |
| --- | --- |
| B1–B3 | Provider connectors, model router, call logging and Tavily integration are implemented. PR #69 persists completed structured attempts before retries. |
| B4 | PR #71 added an opt-in local Docker Python runner. This batch adds generated checks through the Python agent entry point, with a live Super → Docker result. The cloud beta request was submitted; activation is pending; see `generated-checks.md`. |
| B5 | Reference tooling exists. Local + Super A2/A3 datasets now exist with agent-reviewed references. The full three-task, three-model matrix remains open. |
| B6 | This batch measures three models at five size/epoch conditions each, validates duration fits on a preselected condition, and registers accepted fits in the planner. All 15 jobs succeeded; validation errors are 2.46%, 1.21% and 0.27%. See `model-cost-validation.md`. |
| B7 | Sensitivity curves now cover both strengths of faults 6–8 across three seeds. Low learning rate degrades quality; rank and epoch changes do not show substantial consistent degradation here. Pre-intervention top-1 recovery is 3/3 at each strength, but all healthy provider cases also rank the epoch fault first; no 80% population guarantee is established. See `recipe-sensitivity.md`. |
| B8 | Target/control recipe builders and cost matching exist. |
| B9 | The simulation study is complete: 210 local LoRA runs cover seven healthy/fault-strength recipes at 30 seeds. Seed-grouped, recipe-strength and frozen Token Factory transfer evaluations are saved in `b9-final-sweep.md`. The only measured quality failure is low learning rate. The strongest local classifier reaches 96.7% balanced accuracy but falls to 25.0% on historical heldout Token Factory cases, so it is rejected for live use. |
| B10 | Noise measurements and a decision-power audit exist (PR #68). Empirical coverage across task/model/sample-size combinations remains incomplete. |
| B11 | Causal statistics and decision rules are implemented; small-sample limitations are documented in the power audit. |
| B12 | The calibration task is complete with a negative deployment decision. Temperature fitting, reliability plots and ECE/Brier audits exist; a development-fitted temperature worsens all three heldout measures. The live default remains T=1. |
| B13–B15 | Versioned prompts, hypothesis sampling and sourced known-issue search are implemented. One-versus-three sampling results are saved. |
| B16 | Dev signature fitting with smoothing and provenance was added in PR #65. Coverage is limited to observed fault classes. |
| B17–B18 | The planned investigation cycle is implemented: persisted states, budgets, paid-step approval, statuses, verdicts and retraining flow. SQLite reservations and REST/UI resume cover process restarts. [Recipe retrain and Tavily](paid-operation-recovery.md) now retain budget reservations across failures. [SQLite worker ownership](sqlite-investigations.md) now coordinates request claims and approvals across processes. [Aggregate provider training usage is reconciled](provider-cost-reconciliation-20261019.md); invoices, the displayed-total difference and per-call inference remain open. |
| B19–B20 | Baselines and ablations exist. PR #66 saves the current dev comparison and limitations. More fault classes require new evidence. |
| B21 | Public cases were collected in PR #64. Reproduction of their reported failures remains open. |
| B22 | API, SSE, reports and restart handling exist. SQLite state was added in PR #70; #97 adds same-host worker ownership and migration instructions. Distributed hosting remains unsupported. |
| B23 | Investigation UI, case launch, costs and report downloads exist. |
| B24 | PR #72 added token-weighted gradient accumulation and injectable fault 12 with CPU equivalence checks. PR #73 added injectable fault 10 (EOS masked as padding), a persisted label audit and CPU controls. PR #74 added fault 9 (prompt loss), its audit and separate supervised-token counts. PR #75 added fault 11 (double label shifting) and an alignment audit. PR #76 added a persisted trainable-parameter inventory and an early stop for fully frozen models. PR #77 added a first-backward gradient audit. PR #78 exposes bounded label checks to mode-B hypothesis generation through the Python investigation entry point. This package adds REST/UI label audits, opt-in local code-switch interventions and a 60-run synthetic CPU generation experiment; see `mode-b-local-package.md`. B24 remains partial: pretrained-LLM quality, arbitrary-code repair and full REST mode-B investigations are not validated. |
| B25 | Weight probes are wired into live runs and the benchmark. Threshold transfer and rank-fault discrimination need more evidence. |
| B26 | A classifier model publication with validated transfer metrics remains open. |

The PR #79 batch covers B4, B6 and B7. A2/A3 now provide train, validation, check and test splits; B5 still requires the full reference matrix. B4, B6 and B7 now have the requested implementations and saved measurements.
Cloud Sandbox activation is pending; B7 detection measures pre-intervention
ranking and does not claim causal confirmation or a statistically supported 80%
threshold. The protocol and
budget are recorded in [the batch protocol](results/b467-20261019/protocol.md).

## Independent third-batch selection audit

A fresh clone of `origin/main` at `e10f4b5` was audited on October 19, 2026.
Under the exclusions for active B4/B6/B7, B9/B12/B17, dataset-dependent B5/B10
and B26, three unfinished independent plan items were not available. B14, B20
and B22 already meet their implementation requirements. B24 already contains
all four requested fault injections and code checks; its remaining extensions
are not three separate plan items. B21 reproduction and B25 transfer validation
retain their evidence dependencies. The user then authorized three B24 sub-tasks: API/UI label audits, local interventions and a synthetic downstream-quality experiment. All three sub-tasks have implementation and local evidence; B24 remains partial.
See the [selection audit](results/eligibility-audit-20261019/README.md).

## B9/B12/B17 publication snapshot

The [follow-up](next-batch-20261019.md) includes the final 210-run B9 analysis,
development and heldout calibration audits, and spending reservations with
REST/UI approval and resume. B9, B12 and the original B17 plan item now have
measured outcomes. B9's classifier fails frozen provider transfer and stays out
of live diagnosis. B12 keeps T=1 because fitted calibration worsens heldout NLL,
Brier and ECE. SQLite worker ownership is implemented. The provider-cost audit
matches 224 successful jobs to aggregate Usage; invoice and per-call inference
reconciliation remain open.

For the current cross-track queue, publication state, fault-1 semantics and demo
requirements, see [Project status](project-status.md). The original provider-side
silent-truncation assumption is withdrawn for the tested mode-A configuration;
pre-upload `answer_truncation` has measured Kaggle and Token Factory evidence.
