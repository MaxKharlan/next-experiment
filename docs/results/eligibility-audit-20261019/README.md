# Third-batch eligibility audit

Audited origin/main at `e10f4b5ffadd4e88baa1eb984fc7f444dd4e96fd` on
October 19, 2026, in an isolated clone. Three independent, unfinished plan items could not be identified under the
supplied exclusions. The user subsequently authorized three B24 sub-tasks;
see [the package report](../../mode-b-local-package.md). No whole plan item was
newly marked complete.

## Selection in plan order

| Items | Decision and evidence |
| --- | --- |
| B1–B3 | Already implemented: connectors, routing and call logs. |
| B4 | Explicitly reserved by the requesting chat. |
| B5 | Excluded: missing approved A2/A3 task datasets. |
| B6–B7 | Explicitly reserved by the requesting chat; PR #79 is now on main. |
| B8 | Already implemented: matched target/control recipes. |
| B9 | Explicitly reserved by another active chat. |
| B10 | Excluded: full task/model/sample-size coverage needs A2/A3. |
| B11 | Already implemented: causal statistics and decision rules. |
| B12 | Explicitly reserved by another active chat. |
| B13 | Already implemented: versioned prompts. |
| B14 | Already implemented in core/agent/ensemble.py and steps.py. Sampling, class-probability aggregation and catalog filtering have tests. docs/results/dev-hypothesis-sampling-20261018 contains the paired one/three-call experiment. |
| B15–B16 | Already implemented: sourced known issues and fitted signature outcomes. Additional fault coverage requires evidence. |
| B17 | Explicitly reserved by another active chat. |
| B18–B19 | Verdict/retraining and comparison agents already exist. |
| B20 | Already implemented: bench/ablations.py, causal and routing tests, saved dev ablations and routing comparison. Broader generalization is not a new unimplemented ablation. |
| B21 | Source collection exists. Remaining reproduction cannot close the complete item locally: catalog training cases include two B9-owned unresolved reports; original configurations or causal isolation are missing. The accumulation-mechanism candidate alone is a partial result. |
| B22 | Already implemented: creation, SSE and reports in apps/api/main.py; SQLite models in core/db/investigations.py; restart/transaction tests. Migration and multiple workers are additional scope, not missing requirements of the original item. |
| B23 | Investigation UI already exists. |
| B25 | Probe exists. Completing simulation-to-provider threshold validation and rank discrimination requires B9 evidence and appropriate real provider cases. Current documentation reports failed threshold transfer and absence of a useful rank decision. |
| B26 | Explicitly excluded until the active B9 package is validated. |
| B24 | All four requested injection mechanisms and code checks exist in jobs/kaggle/train_lora.py, core/training and tests. REST/UI attachment support, mode-B interventions and downstream quality studies are remaining extensions, not three separate plan items. |

B24 appears after B26 in the original plan and was assessed in that order.
Existing status limitations remain valid; this audit does not turn a mechanism
test into a reproduced external quality failure or mark B25 complete.

## Verification

Commands run from the isolated repository root:

```bash
uv run ruff check .
uv run --group ml --group eval pytest -q
```

Results are recorded in `validation.txt`. No API inference, provider training,
Tavily request or paid experiment was submitted. New service spending: USD 0.
The original checkout and its artifacts were not used. No commit, push or PR
was created.
