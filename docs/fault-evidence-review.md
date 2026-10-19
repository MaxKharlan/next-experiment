# Fault evidence review after the length-limit experiment

Reviewed against main `9258ae5` (PR #86) on 19 October 2026. This review separates
an observed mechanism, a quality loss and an intervention benefit. A setting or
probe signature alone does not establish all three. No new training or paid
inference was run for this follow-up.

## Catalog and outstanding experiments

| Fault | What is supported | What remains to establish |
|---|---|---|
| 1, mode A: `answer_truncation` | Pre-upload answer damage and partial recovery measured on BANKING77, Kaggle and Token Factory; [results](answer-cut.md) | [Clean-prefix probe controls measured](answer-cut-specificity-controls-20261019.md); model-quality effects, other tasks/models and a fresh complete agent trace |
| 1, mode B: `truncation` | Length probe detects over-limit examples; trainer code can remove tokens | Observe actual answer-token removal, measure quality loss, compare the repair with a specified control; length alone is insufficient |
| 2: chat-template mismatch | Catalog hypothesis; token counting uses a template | Freeze training and inference templates, demonstrate a consequential mismatch and repair; differing template text can be harmless |
| 3: contradictory labels | Catalog hypothesis | Establish that inputs require equivalent answers, distinguish valid ambiguity, implement observed-data detection and a repair/control pair |
| 4: train/validation overlap | Exact row and prompt audit exists | Measure optimistic validation and quality on a clean check split; implement near-duplicate detection and intervention. Overlap alone need not degrade the model |
| 5: distribution shift | Catalog hypothesis | Define the intended production distribution, measure the shift and its task-quality effect; add a repair/control comparison |
| 6: low learning rate | Large quality loss in the local sensitivity study; historical intervention traces | Fresh task/model evaluation and competing-cause specificity; a larger rank can compensate for low learning rate |
| 7: too many epochs | Recipe intervention exists; measured 9/18-epoch sensitivity is not consistently harmful | Find a reproducible harmful setting before evaluating fault detection; epochs alone are not an overfitting label |
| 8: small LoRA rank | Recipe intervention exists; local rank 2/4 barely changes mean quality | Demonstrate a capacity-limited case; rank/alpha changes need resource accounting and competing explanations |
| 9: prompt loss | Label audit and explicit code switch; synthetic quality experiment found no benefit from repair | Pretrained-LLM degradation and repair, including clean references with intentionally supervised prompts |
| 10: EOS masking | Label audit and synthetic GRU quality benefit | Pretrained-LLM confirmation; distinguish EOS from actual padding positions |
| 11: double label shift | Alignment audit and synthetic GRU quality benefit | Pretrained-LLM confirmation and trainer-specific shift convention |
| 12: accumulation normalization | Code switch and gradient tooling; synthetic quality experiment found no benefit | A task with consequential variable token counts and pretrained-LLM repair evidence |

Sources: [recipe sensitivity](recipe-sensitivity.md),
[mode-B local package](mode-b-local-package.md),
[data fault builder](../bench/faults/data.py),
[dataset audit](../core/eval/audit_case.py),
[catalog](../core/agent/catalog.py). Faults 2–5 remain hypotheses without measured
repair pairs. Label references for mode-B audits are explicitly supplied by the
caller; the audits do not certify their semantic correctness.

## Corrections implemented in this follow-up

New intervention events carry `comparison_evidence` with version
`intervention-evidence-v1`. It records seed counts, minimum effect, the 95% interval
method and supplied noise SD, permutation alpha and minimum attainable p, and
whether the permutation gate was applied. At two seeds per arm the historical
rule skips the gate because the minimum p is 1/6. The calculation and its legacy
`confirmed` / `rejected` / `inconclusive` outcomes are unchanged.

This metadata describes the comparison's scope. It does not certify noise
transfer, resource matching, a unique cause or recovery to healthy quality.
The live screen now shows these limits and the p-value. Legacy records without
the metadata are shown as having unrecorded settings; the renderer does not infer
settings from their scores. Saved JSON, older prompts and measured demo inputs
retain their provenance.

The hypothesis board now reads the recorded posterior from intervention events;
previously it kept the probabilities from the last probe. Its labels describe
support for the tested fix or an effect below the threshold, without striking out
the underlying cause. Legacy events without a posterior retain the last recorded
probabilities.

The new `verdict-v3` prompt receives the same metadata and no longer assumes that
the control costs at least as much. Its wording is unmeasured: no improvement in
agent accuracy is claimed. Hypotheses-v4 and interpret_probe-v2 remain current.

Mode B now needs an explicit trainer `context_length` for the length probe. It
cannot silently inherit Token Factory's 8192 default. New continuations record
their investigation mode; switching modes is rejected. Frozen or resumed
hypotheses also pass the catalog boundary before any model or training call.
Legacy mode-B `truncation` reports remain readable, but cannot become executable
mode-A investigations simply by loading their hypothesis list.

## Next work and ownership

1. B9 finished all 210 runs and published grouped analysis in PR #91. Frozen
   Token Factory transfer failed; both classifiers stay offline. Do not repeat
   this sweep. See [the completed analysis](b9-final-sweep.md).
2. Prepare fresh data/protocol artifacts for the answer-cut agent trace and
   the task/model matrix using [the fresh evaluation protocol](fresh-evaluation-protocol.md).
   Historical check/heldout reuse cannot establish generalization.
3. [Clean prefix-label and alternative JSON-malformation controls](answer-cut-specificity-controls-20261019.md)
   are now measured: observationally identical labels can be clean or damaged.
   Benign recipe controls and independent likelihood calibration remain open.
   Current signature values are heuristic; keep classifiers offline and T=1.
4. Define measured mechanisms and repair/control pairs for faults 2–5, then
   pretrained-LLM confirmation for 9–12. These remain open work, not closed by the
   catalog or the local audit UI.
5. Reconcile invoices and extend budget ownership to retrain, Tavily and multiple
   workers. A recorded training estimate is not billed cost. Submission/video and
   eligibility remain separate user-owned work.
