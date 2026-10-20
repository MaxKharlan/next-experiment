# Qwen reference quality — 20 October 2026

Qwen3-0.6B completed its Cloud evaluation at 17:29:54 UTC. The collector verified
all 2,506 saved answers, 21 score files, twelve adapter validation scores and
three selected-adapter check scores. Qwen3-1.7B is still running; its results
will be reported after collection and verification.

| Task | Base check | Selected adapter | Selected validation | Selected check |
| --- | --- | --- | --- | --- |
| Banking77 | 0/154 | g3: learning rate 0.0002, six epochs | 104/154 (67.53%) | 104/154 (67.53%) |
| Function calling | 0/102 | g0: learning rate 0.0001, three epochs | 102/102 (100%) | 100/102 (98.04%) |
| JSON extraction | 0/102 | g0: learning rate 0.0001, three epochs | 102/102 (100%) | 102/102 (100%) |

Selection used all four validation scores for each task, with grid index as
the declared tie-breaker. Only the selected adapter was evaluated on check.
All four function-calling adapters tied at 100% validation; the first three
JSON adapters tied at 100%, while g3 scored 99/102. Banking validation scores
were 95/154, 103/154, 95/154 and 104/154 for g0 through g3.

The base model's zero scores need context. Every saved base answer for both
structured tasks starts with a Markdown code fence: 204/204 answers per task
across validation and check. The frozen metrics require a bare JSON object and
reject these answers. The zero therefore does not establish that their enclosed
field values are wrong. On Banking77, the base often returned a generic label:
53/154 check answers were the literal word `intent`; none matched its target
label. The [format observations](results/qwen-reference-cloud-20261020/Qwen3-0.6B/base-format-observations.json)
retain these counts without changing the metrics or rerunning generation.

These are development references on one training seed. The structured splits
were previously inspected. They do not establish independent generalization,
fault recovery or completion of the original three-model matrix. No final test
was evaluated. The [prospective CUDA protocol](results/qwen-reference-cloud-20261020/Qwen3-0.6B-protocol.json)
records the move from MPS before generation; equivalence of these trained
adapters across MPS and CUDA remains unverified.

The exported [analysis](results/qwen-reference-cloud-20261020/Qwen3-0.6B/analysis.json)
and [raw answers](results/qwen-reference-cloud-20261020/Qwen3-0.6B/evaluation/scores)
preserve all successes and failures. Verification checked source and data hashes,
adapter bindings, runtime versions, output hashes, row identities, recomputed
metrics and selection. The [terminal receipt](results/qwen-reference-cloud-20261020/Qwen3-0.6B/terminal.json)
records provider status `COMPLETED`. Its $6 reservation stays unresolved until
billing; completion is not an invoice.
