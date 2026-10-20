# A10 lexical overlap candidates

A new offline audit lists exact-normalized and lexically similar train/validation
prompts for review. On 22 authored controls it detects 7 repeated requests,
misses 4 paraphrases and flags 4 distinct requests incorrectly. It correctly
rejects 7 distinct requests. This is a lexical baseline for A10; semantic
validation and fault-4 quality measurements remain open.

## Running the audit

```bash
python -m scripts.audit_lexical_overlap \
  --train /path/to/train.jsonl --validation /path/to/validation.jsonl \
  --out /new/output/overlap.json
```

Both inputs use the repository's chat JSONL format and must be different files.
The command reads only the explicitly supplied files, records their SHA-256
hashes, and refuses to overwrite its output. It accepts no check/test argument.
It is a standalone audit; it adds no live diagnostic signature or likelihood.

The output contains every candidate's original train/validation line numbers,
match type, similarity and exact reference-answer agreement. Final answers do
not affect detection. The validation candidate share counts each validation row
once, even when it matches several training rows. Blank lines do not shift the
reported source line numbers. No rows are removed or rewritten.

## Fixed method

Prompt messages are normalized with Unicode NFKC, case folding and collapsed
whitespace. Identical normalized prompts are exact candidates. Fuzzy candidates
require the same message-role sequence and identical normalized system messages.
The representation is a set of character 5-grams tagged by message index,
excluding system-message text. Both prompts must contain at least 40 non-system
characters. Jaccard similarity must be at least 0.85.

The threshold, representation, context guard and minimum length were chosen
before measuring the controls. There is no threshold sweep, fitted embedding or
label-based filtering. Role/message positions remain distinct across turns.

The implementation exhaustively checks train × validation pairs, up to 500,000
by default. Exceeding that cap raises an error; it never reports a sampled search
as clean. `--max-pairs` accepts an explicit positive limit for larger files.
Runtime depends on both pair count and prompt length; this baseline is intended
for small audits and has no retrieval index.

## Constructed controls

The [control definitions](../bench/controls/lexical-overlap-v1.json) contain 22
newly authored request pairs with author-assigned same-request labels. Identical
intent categories alone do not make two requests duplicates: an amount, entity
or negation can change the requested action. These labels have not received an
independent human review and do not enter scored benchmark truth.

| Control family | Pairs | Candidates detected | Interpretation |
| --- | ---: | ---: | --- |
| Normalized exact | 3 | 3 | Repeats found |
| Punctuation/surface variants | 4 | 4 | Repeats found |
| Paraphrases | 4 | 0 | All missed |
| Changed number | 2 | 2 | Both false positives |
| Changed entity | 1 | 0 | Correct rejection |
| Negation | 2 | 2 | Both false positives |
| Different system context | 2 | 0 | Correct rejections |
| Short distinct requests | 2 | 0 | Correct rejections |
| Unrelated requests | 2 | 0 | Correct rejections |

The [published report](results/lexical-overlap-controls-20261020-v2/report.json)
retains every pair and error. The constructed confusion counts are TP=7, FP=4,
TN=7, FN=4. They are not a deployment precision/recall estimate. Similarity can
ignore consequential changes in long prompts; low word overlap can hide
paraphrases. Case folding may also merge case-sensitive identifiers. A candidate
therefore needs content review, even when its normalized text is identical.

The [initial bundle](results/lexical-overlap-controls-20261020/protocol.json)
remains saved with its runner source. Revision 2 only changes the exception type
for a non-boolean control label from ValueError to TypeError to satisfy lint;
all inputs, detection logic and thresholds remain identical. Both reports are
byte-identical. These are deterministic CPU text comparisons, with no provider
calls, trained-model evaluations or training runs.

Verify the current frozen bundle:

```bash
python -m bench.lexical_overlap_controls verify \
  --out docs/results/lexical-overlap-controls-20261020-v2
```

The verifier checks input/source hashes and reconstructs all candidate results.
Unit tests also cover wrong context, source line numbers, ambiguous numeric
changes, duplicate line IDs, search limits and evidence tampering.

## Remaining fault-4 work

A10 still needs independent semantic labels and a measured semantic method,
including false-positive review. Fault 4 also needs a deterministic overlap
injection, declared deduplication and matched control, and model-quality evidence
comparing validation optimism with a clean check. Candidate similarity alone
does not establish contamination provenance or a harmful training failure.
