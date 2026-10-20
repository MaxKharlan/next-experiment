# Completed REST code investigation

Investigation `inv-1f6c0581` completed through the REST API. It returned HTTP 202,
streamed six saved events, produced a report and returned the same report after
the API restarted. Repeating the POST with the original idempotency key reused
the investigation without launching another run.

The study compared restored EOS supervision with the observed `eos_as_pad`
implementation. Each arm trained Qwen3-0.6B adapters with seeds 4201–4204 on 96
training examples, with 32 validation and 64 check examples. All eight runs used
the frozen recipe, 36 optimizer steps and MPS. Their adapter hashes are distinct.

| Seed | Restored EOS supervision | Observed implementation |
| --- | ---: | ---: |
| 4201 | 55/64 | 55/64 |
| 4202 | 57/64 | 56/64 |
| 4203 | 56/64 | 56/64 |
| 4204 | 56/64 | 56/64 |
| Mean exact match | 87.50% | 87.11% |

The recorded comparison rejected this repair under its frozen rule; the agent
stopped at its intervention limit with an `inconclusive` diagnosis. The difference
was 0.39 percentage points, with a 95% interval from −1.44 to 2.22 points and
permutation p=0.5. This reused development task supplies no independent recovery
or unique-cause claim.

The single Nemotron Super call recorded 1,555 input and 808 output tokens, with a
$0.0011937 cost estimate and provider request IDs. Training ran locally.

Verification checked source and dataset hashes, recipes, seeds, training audits,
saved scores, the recomputed comparison, SSE/report agreement and restart
receipts. The frozen executor saved aggregate check scores; it did not save
individual generated answers. No evaluation was repeated for this export.

[Evidence and hashes](results/rest-code-investigation-20261020/verification.json)
and the [file manifest](results/rest-code-investigation-20261020/manifest.json)
accompany the report, SSE stream and eight measurement directories.
`scripts.export_rest_code_trace` reproduces the verification from local artifacts.
