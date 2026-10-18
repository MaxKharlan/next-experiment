# Track B reconciliation

Compared the original project plan with `main` through PR #74
(`d85b3bc`) and the repository implementation on October 18, 2026.
Checkboxes in the original plan do not establish that every acceptance condition
has measured evidence.

| Plan items | Existing implementation and remaining work |
| --- | --- |
| B1–B3 | Provider connectors, model router, call logging and Tavily integration are implemented. PR #69 persists completed structured attempts before retries. |
| B4 | PR #71 added an opt-in local Docker Python runner. Agent integration and Token Factory Sandbox beta access remain open; see `docker-checks.md`. |
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
| B22 | API, SSE, reports and restart handling exist. SQLite state was added in PR #70; migration and multiple workers remain unsupported. |
| B23 | Investigation UI, case launch, costs and report downloads exist. |
| B24 | PR #72 added token-weighted gradient accumulation and injectable fault 12 with CPU equivalence checks. PR #73 added injectable fault 10 (EOS masked as padding), a persisted label audit and CPU controls. PR #74 added fault 9 (prompt loss), its audit and separate supervised-token counts. This change adds fault 11 (double label shifting) and an alignment audit. Agent-facing checks, trainable-parameter inspection and downstream quality evidence remain open. |
| B25 | Weight probes are wired into live runs and the benchmark. Threshold transfer and rank-fault discrimination need more evidence. |
| B26 | A classifier model publication with validated transfer metrics remains open. |

The selected task is the fault-11 double-label-shift mechanism in B24. It adds
pre-shifted training labels and records their alignment against healthy targets.
See [Double label shift](double-label-shift.md). No Kaggle job, downloaded model
or paid training was used. B24 remains partial despite injection coverage for 9–12.
