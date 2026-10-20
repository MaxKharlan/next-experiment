---
language:
  - en
license: other
license_name: apache-2.0-and-cc-by-4.0
license_link: https://huggingface.co/datasets/MakHH/FT-Fail/blob/main/LICENSES.md
task_categories:
  - text-classification
tags:
  - fine-tuning
  - evaluation
  - banking77
  - negative-results
size_categories:
  - n<1K
configs:
  - config_name: cases
    default: true
    data_files:
      - split: dev
        path: cases/dev.jsonl
      - split: historical_heldout
        path: cases/heldout.jsonl
  - config_name: methods
    data_files:
      - split: dev
        path: methods/dev.jsonl
      - split: historical_heldout
        path: methods/heldout.jsonl
  - config_name: semantic_audit
    data_files:
      - split: dev
        path: semantic_audit/dev.jsonl
  - config_name: semantic_reference
    data_files:
      - split: dev
        path: semantic_reference/dev.jsonl
---

# FT-Fail: saved fine-tuning diagnosis cases

FT-Fail contains evidence from the [Next Experiment](https://github.com/MaxKharlan/next-experiment)
project. This release packages completed measurements. It contains 13 historical
Token Factory cases, their 52 method score rows, and a separate 200-query automatic
label audit with 400 reference queries.

## Configurations

| Configuration | Rows | Content |
| --- | ---: | --- |
| `cases` | 5 dev + 8 historical held-out | Symptoms, recipe, checkpoint losses, planted faults and saved quality |
| `methods` | 20 dev + 32 historical held-out | Saved diagnosis, ranking, fix, cost and failure scores for four methods |
| `semantic_audit` | 200 dev | Query, observed label, source label and recorded corruption |
| `semantic_reference` | 400 dev | Separate reference queries with original labels |

The case input is stored in `case_json`; the planted truth stays in
`injected_faults`. Evaluate a diagnosis using only the case input. Local artifact
paths in the original manifests identify historical files and are not download
links. Model checkpoints and all original training artifacts are outside this
release.

The historical held-out run finished before publication. Its evidence is already
published in the repository and is included here without rerunning methods.
These cases are now public and cannot serve as an untouched final evaluation
for further development. No pending experiment's final split is included.

## Findings and limits

The historical benchmark covers BANKING77, Qwen3-0.6B and predominantly low
learning-rate faults. It is too small to establish performance across tasks,
models or fault classes. A beneficial repair need not identify the original
cause: increasing rank also improved several low-learning-rate cases.
The [benchmark report](evidence/docs/benchmark.md) describes the fixed decision
rule, failed repairs and method comparisons.

The automatic semantic audit uses eight intents from BANKING77's training split.
Of its 200 queries, 100 retain their labels, 50 receive a different valid label,
and 50 receive an invalid JSON-wrapped label. The original human-review requirement
was replaced by an explicitly recorded automatic-reference acceptance amendment.
The labels are source labels plus known modifications; they are not independent
human judgments.

Nano's explicit contradiction flag found none of the 50 wrong labels. Super found
31, with two HTTP failures retained. Neighbor and cleanlab comparisons, including
false positives, are documented in the
[semantic report](evidence/docs/semantic-audit-20261020.md).
The fixed semantic duplicate threshold missed all four authored paraphrase
controls. These measurements do not justify automatic data deletion or calibrated
production fault probabilities.

## Provenance and reuse

`manifest.json` records SHA-256 hashes and the source repository revision.
The `evidence/` directory includes original case manifests, protocols, acceptance
amendment and reports. Packaging does not issue new model calls or change scores.
The code that constructs this release is `scripts/package_ft_fail.py` in the
source repository.

BANKING77 queries come from PolyAI's task-specific dataset, pinned to commit
`57ec275d8078af65b7731c2a98be812d844a6d6b` for the automatic audit. See
[the dataset card](https://huggingface.co/datasets/PolyAI/banking77) and
[Casanueva et al. (2020)](https://aclanthology.org/2020.nlp4convai-1.5/).
The automatic audit records original source line numbers. Changes include query
selection, label replacement and label wrapping; query text is retained.

See [LICENSES.md](LICENSES.md) for source attribution and the separate terms for
BANKING77 text and project-authored code, manifests and measurements.
