# Track B reconciliation

Compared the original project plan with `main` through PR #79
(`e10f4b5`) and the repository implementation on October 19, 2026.
Checkboxes in the original plan do not establish that every acceptance condition
has measured evidence.

| Plan items | Existing implementation and remaining work |
| --- | --- |
| B1–B3 | Provider connectors, model router, call logging and Tavily integration are implemented. PR #69 persists completed structured attempts before retries. |
| B4 | PR #71 added an opt-in local Docker Python runner. This batch adds generated checks through the Python agent entry point, with a live Super → Docker result. The cloud beta request was submitted; activation is pending; see `generated-checks.md`. |
| B5 | Reference tooling exists. Approved A2/A3 datasets are absent, so the full three-task, three-model matrix remains open. |
| B6 | This batch measures three models at five size/epoch conditions each, validates duration fits on a preselected condition, and registers accepted fits in the planner. All 15 jobs succeeded; validation errors are 2.46%, 1.21% and 0.27%. See `model-cost-validation.md`. |
| B7 | Sensitivity curves now cover both strengths of faults 6–8 across three seeds. Low learning rate degrades quality; rank and epoch changes do not show substantial consistent degradation here. Pre-intervention top-1 recovery is 3/3 at each strength, but all healthy provider cases also rank the epoch fault first; no 80% population guarantee is established. See `recipe-sensitivity.md`. |
| B8 | Target/control recipe builders and cost matching exist. |
| B9 | Local LoRA runs and a frozen loss classifier with seed-grouped evaluation were added in PR #67. Hundreds of runs and a validated transfer result are still absent. |
| B10 | Noise measurements and a decision-power audit exist (PR #68). Empirical coverage across task/model/sample-size combinations remains incomplete. |
| B11 | Causal statistics and decision rules are implemented; small-sample limitations are documented in the power audit. |
| B12 | Calibration audit, temperature fitting and reliability plots exist. Independent validation and calibrated live priors remain separate work. |
| B13–B15 | Versioned prompts, hypothesis sampling and sourced known-issue search are implemented. One-versus-three sampling results are saved. |
| B16 | Dev signature fitting with smoothing and provenance was added in PR #65. Coverage is limited to observed fault classes. |
| B17–B18 | Persisted steps, usage, probe budget checks, approvals, verdicts and retraining flow exist. Cross-process execution and hard spending reservations remain open. |
| B19–B20 | Baselines and ablations exist. PR #66 saves the current dev comparison and limitations. More fault classes require new evidence. |
| B21 | Public cases were collected in PR #64. Reproduction of their reported failures remains open. |
| B22 | API, SSE, reports and restart handling exist. SQLite state was added in PR #70; migration and multiple workers remain unsupported. |
| B23 | Investigation UI, case launch, costs and report downloads exist. |
| B24 | PR #72 added token-weighted gradient accumulation and injectable fault 12 with CPU equivalence checks. PR #73 added injectable fault 10 (EOS masked as padding), a persisted label audit and CPU controls. PR #74 added fault 9 (prompt loss), its audit and separate supervised-token counts. PR #75 added fault 11 (double label shifting) and an alignment audit. PR #76 added a persisted trainable-parameter inventory and an early stop for fully frozen models. PR #77 added a first-backward gradient audit. PR #78 exposes bounded label checks to mode-B hypothesis generation through the Python investigation entry point. This package adds REST/UI label audits, opt-in local code-switch interventions and a 60-run synthetic CPU generation experiment; see `mode-b-local-package.md`. B24 remains partial: pretrained-LLM quality, arbitrary-code repair and full REST mode-B investigations are not validated. |
| B25 | Weight probes are wired into live runs and the benchmark. Threshold transfer and rank-fault discrimination need more evidence. |
| B26 | A classifier model publication with validated transfer metrics remains open. |

The PR #79 batch covers B4, B6 and B7. B5 is deferred until approved A2/A3
train, validation and held-out data exists. B4, B6 and B7 now have the requested implementations and saved measurements.
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
