# Track B reconciliation

Compared the original project plan with `main` through PR #69
(`f1887f9`) and the repository implementation on October 18, 2026.
Checkboxes in the original plan do not establish that every acceptance condition
has measured evidence.

| Plan items | Existing implementation and remaining work |
| --- | --- |
| B1–B3 | Provider connectors, model router, call logging and Tavily integration are implemented. PR #69 persists completed structured attempts before retries. |
| B4 | `jobs/sandbox/` is empty. Isolated execution of generated checks remains open. |
| B5–B6 | Reference and cost tooling exists. Saved evidence does not cover the full planned three-task, three-model matrix. |
| B7–B8 | Fault injection and target/control pairs exist. Evidence of task-quality degradation is concentrated on low learning rate; rank and epoch sensitivity need further experiments. |
| B9 | Local LoRA runs and a frozen loss classifier with seed-grouped evaluation were added in PR #67. Hundreds of runs and a validated transfer result are still absent. |
| B10 | Noise measurements and a decision-power audit exist (PR #68). Empirical coverage across task/model/sample-size combinations remains incomplete. |
| B11 | Causal statistics and decision rules are implemented; small-sample limitations are documented in the power audit. |
| B12 | Calibration audit, temperature fitting and reliability plots exist. Independent validation and calibrated live priors remain separate work. |
| B13–B15 | Versioned prompts, hypothesis sampling and sourced known-issue search are implemented. One-versus-three sampling results are saved. |
| B16 | Dev signature fitting with smoothing and provenance was added in PR #65. Coverage is limited to observed fault classes. |
| B17–B18 | Persisted steps, usage, probe budget checks, approvals, verdicts and retraining flow exist. Cross-process execution and hard spending reservations remain open. |
| B19–B20 | Baselines and ablations exist. PR #66 saves the current dev comparison and limitations. More fault classes require new evidence. |
| B21 | Public cases were collected in PR #64. Reproduction of their reported failures remains open. |
| B22 | API, SSE, reports and restart handling exist. SQLite state is implemented in this change; migration and multiple workers remain unsupported. |
| B23 | Investigation UI, case launch, costs and report downloads exist. |
| B24 | Mode B fault classes 9–12 remain future work. |
| B25 | Weight probes are wired into live runs and the benchmark. Threshold transfer and rank-fault discrimination need more evidence. |
| B26 | A classifier model publication with validated transfer metrics remains open. |

The selected task is the missing SQLite persistence portion of B22. It uses
local API tests and does not depend on the pending Kaggle answer-cut experiment.
See [SQLite investigation state](sqlite-investigations.md) for activation and
limits. This change does not repeat the B9, B10, B16 or B20 experiments.
