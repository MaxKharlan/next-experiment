# B21: public fine-tuning reports

Five reports were selected through four Tavily basic searches and direct reading
of issue discussions and forum replies. Discovery used four Tavily credits;
no training or model inference was run during discovery. The [catalog](../bench/public_cases/catalog.json)
records symptoms, proposed causes, evidence links, missing inputs and reproduction
plans. [Discovery metadata](../bench/public_cases/discovery.json) records queries
and returned URLs. Summaries are paraphrases; source datasets and discussion
contents are not redistributed.

| Report | Evidence | Next owner |
|---|---|---|
| [TRL #696](https://github.com/huggingface/trl/issues/696) | Author reports a padding-token workaround restoring EOS behavior | A25: label/token inspection and reproduction |
| [Transformers #38837](https://github.com/huggingface/transformers/issues/38837) | Historical CPU run confirms half-gradient at accumulation 2; accumulation-3 controls fail | B21/B24: partial mechanism evidence |
| [HF #68899](https://discuss.huggingface.co/t/mistralai-mistral-7b-instruct-v0-2-fine-tuning-prompt-format/68899) | Formatting symptoms persist after suggested preprocessing | A25: unresolved data/formatting candidate |
| [HF #126058](https://discuss.huggingface.co/t/bad-performance-finetuning-llama-chat-and-instruct-models-on-gsm8k/126058) | Pipeline performance difference; no isolated cause | B9: evaluation and training comparison |
| [TRL #3910](https://github.com/huggingface/trl/issues/3910) | Proposed explanations disputed; original setup unavailable | B9: retain as unresolved |

The collection completes the source-gathering part of B21. Reproducing the
training cases remains work for track B; data cases feed A25. None of these
reports has been fully reproduced by this project or added to scored benchmark cases.
The [historical Trainer experiment](transformers-38837-reproduction-20261020.md)
provides partial mechanism evidence for #38837: four of six case/rate comparisons
match the analytic reference, while both accumulation-3 controls fail. Its status
is `stopped`; original-case and downstream-quality reproduction remain open.
A closed issue is not evidence of a confirmed diagnosis. The accumulation issue
is a training-mechanism lead, not yet a demonstrated silent quality failure.
These historical reports also do not establish bugs in current Token Factory.

## Admission to the benchmark

A reproduction must record package versions, model revision, seed, data provenance
and redistribution rights. Verify the symptom against a healthy reference,
then compare the proposed fix with a matched control on separate evaluation data.
Persist the run IDs, metrics and manifests. Only after that evidence exists should
an independently reviewed truth label enter `bench/cases`.

Keep unresolved reports available for testing abstention; do not convert a
suggested cause into a label. Use synthetic non-sensitive data when the original
examples cannot be redistributed. Such experiments reproduce a mechanism and
must not be presented as exact reproductions of the original user's run.

## Remaining reproduction work

For TRL #696, inspect whether EOS positions survive label masking before spending
on training. The frozen #38837 experiment retains its failed controls; any new
diagnosis needs a separate protocol. The remaining reports need more evidence
before paid runs.

The [benchmark failure inventory](failure-audit.md) covers our saved model results
and is a separate artifact; it does not replace this external-source collection.
