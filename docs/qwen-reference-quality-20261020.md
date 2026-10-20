# Qwen reference quality — 20 October 2026

Both Qwen models completed their Cloud evaluations: 0.6B at 17:29:54 UTC and
1.7B at 18:07:04 UTC. The collector verified 5,012 saved answers across 42 score
files: per model, twelve adapter validation scores, three selected-adapter
check scores and six base scores.

| Model | Task | Base check | Selected adapter | Selected validation | Selected check |
| --- | --- | --- | --- | --- | --- |
| 0.6B | Banking77 | 0/154 | g3 | 104/154 (67.53%) | 104/154 (67.53%) |
| 0.6B | Function calling | 0/102 | g0 | 102/102 (100%) | 100/102 (98.04%) |
| 0.6B | JSON extraction | 0/102 | g0 | 102/102 (100%) | 102/102 (100%) |
| 1.7B | Banking77 | 1/154 (0.65%) | g3 | 112/154 (72.73%) | 115/154 (74.68%) |
| 1.7B | Function calling | 102/102 (100%) | g0 | 102/102 (100%) | 102/102 (100%) |
| 1.7B | JSON extraction | 102/102 (100%) | g0 | 102/102 (100%) | 102/102 (100%) |

Recipe g0 uses learning rate 0.0001 and three epochs; g3 uses learning rate
0.0002 and six epochs. Each model selected the same recipe index per task.
For 1.7B, both structured tasks already score 100% with the base model; the
selected adapters show no improvement on these checks.

Selection used all four validation scores for each task, with grid index as
the declared tie-breaker. Only the selected adapter was evaluated on check.
For 0.6B, all four function-calling adapters tied at 100% validation; the first
three JSON adapters tied at 100%, while g3 scored 99/102. Banking validation
scores were 95/154, 103/154, 95/154 and 104/154 for g0 through g3.
For 1.7B, every structured adapter scored 100% validation; Banking validation
scores were 94/154, 103/154, 110/154 and 112/154.

The 0.6B base model's zero scores need context. Every saved base answer for both
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

The exported analyses ([0.6B](results/qwen-reference-cloud-20261020/Qwen3-0.6B/analysis.json),
[1.7B](results/qwen-reference-cloud-20261020/Qwen3-1.7B/analysis.json)) and raw answers
([0.6B](results/qwen-reference-cloud-20261020/Qwen3-0.6B/evaluation/scores),
[1.7B](results/qwen-reference-cloud-20261020/Qwen3-1.7B/evaluation/scores))
preserve all successes and failures. Verification checked source and data hashes,
adapter bindings, runtime versions, output hashes, row identities, recomputed
metrics and selection. Both terminal receipts
([0.6B](results/qwen-reference-cloud-20261020/Qwen3-0.6B/terminal.json),
[1.7B](results/qwen-reference-cloud-20261020/Qwen3-1.7B/terminal.json))
record provider status `COMPLETED`. Each $6 reservation stays unresolved until
billing; completion is not an invoice.
