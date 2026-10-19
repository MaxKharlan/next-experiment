# Structured tasks and saved reports

A2 extracts ticket ID, product, priority, callback preference and due date as strict JSON. A3 selects track_order, cancel_order or schedule_pickup and validates the exact argument schema. Duplicate JSON keys, extra fields, type coercion and invalid calendar dates are rejected. Exact match ignores object key order; secondary metrics record JSON validity, schema validity and field/function-name accuracy.

Each task contains 696 examples: 390 train, 102 validation, 102 check and 102 test. Of the 1,392 total, 48 are frozen Nemotron Super paraphrases and 1,344 are deterministic local examples. Entity IDs are disjoint across splits; semantic templates are shared. This is a synthetic integration dataset, not evidence of broad generalization. Files, schemas and generator have recorded SHA-256 hashes.

The agent checked every reference using a separate bounded text parser and read all 48 Super paraphrases. All pairs agree. Human review was not performed: the user requested agent review instead. The earlier question incorrectly called T-100904 a router; both saved request and reference correctly say printer. No dataset correction was needed.

A real Super inference smoke test scored 12/12 exact matches (six per task, combining local and Super wording). It does not validate fine-tuned Qwen quality or the full B5 matrix. Generation cost was $0.0068784 and smoke cost $0.0029841, estimated from repository prices, including one rejected generation response. Raw outputs and usage are saved in docs/results/structured-super-20261019 and structured-smoke-20261019.

The evaluator is connected to InvestigationCase, API case preparation, paired checks and retraining. Existing model restrictions remain in force. POST /api/investigations/audit-case persists a free dataset audit; the live UI exposes it beside case validation. GitHub Pages replays saved reports and cannot run the API.

## Reproduction

Run from the repository root with a new output directory:

```sh
uv run python -m bench.tasks.structured --out artifacts/rebuilt-tasks --super-data docs/results/structured-super-20261019/augmentation.json
uv run python -m scripts.review_structured --root artifacts/rebuilt-tasks --out artifacts/rebuilt-review
uv run --group ml --group eval python -m scripts.build_report_catalog --artifacts artifacts/rebuilt-catalog
uv run python -m scripts.build_pages --out _site
```

The site contains 34 saved reports: the existing confirmed fault-6 experiment, the existing B24 label audit, 24 historical measured traces, four fresh code audits and four structured dataset audits. The historical traces preserve their inconclusive results. Local audits show observed mechanisms or data defects and never claim downstream causal confirmation. Faults 2–5 remain uncovered because no measured implementation reports were available. The UI groups audits, confirmed results and unconfirmed traces separately.

A2/A3 implementation and agent review are delivered. Original human sign-off, full training evaluations and B5's three-task/three-model matrix remain open. B24 retains its separately documented partial status.
