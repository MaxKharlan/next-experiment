# Fault 1: answer damage and length limits

The original mode-A plan assumed that Token Factory silently truncated long
examples. The saved tests on October 18 contradicted that assumption: a context
length of 256 was rejected at job creation; an example of 9013 tokens with an
8192 limit failed dataset processing before training. This evidence applies to
the tested configuration; it is not a guarantee about every provider or future
version. See [the recorded experiment](answer-cut.md#why-this-fault).

The mode-A silent fault is now `answer_truncation`: user-side preparation cuts
assistant answers before upload. The separate class `truncation` means a custom
mode-B trainer removes tokens at its sequence limit. The catalog already kept
these classes separate before this follow-up. New hypothesis steps record
`catalog_version: answer-cut-v1`; hypotheses-v4, interpret_probe-v2 and verdict-v2
make the evidence boundaries explicit. Earlier prompt files and saved reports
retain their identifiers and contents. Do not relabel historical `truncation`
records as `answer_truncation`.

## A8 length preflight

`ExampleLength` counts rendered chat-template tokens. It cannot tell whether a
trainer actually cut tokens, nor detect answers already shortened before upload.
In mode A, over-limit input means a risk of explicit rejection. In mode B, inspect
the trainer's token/label handling before attributing quality loss to truncation.

New raw reports carry `semantics_version: length-preflight-v1`, `mode`,
`over_limit_found` and an interpretation. `over_limit_found` detects any offending
train or validation example. The legacy categorical outcome `truncated` still
means at least 1% of training examples exceeded the limit; it is retained for
stored signatures and replay compatibility. Thus an `ok` categorical outcome is
not an upload approval: one offending example can still fail a job. The UI uses
the raw preflight finding for new reports. Signature likelihoods remain heuristic.

## Current implementation and evidence

- `bench/faults/data.py` damages train/validation deterministically and leaves
  check/test unchanged. Fault manifests are evaluation ground truth.
- `AnswerCut` reads observed answers and flags suspicious prefixes or broken JSON.
  Legitimate prefix labels remain a source of false positives.
- `answer_cut_pair` drops probe-flagged rows for the target and the same number of
  random unflagged rows for the control. It receives no hidden clean answers or
  planted-fault manifest. The measured control trained about 9% more tokens;
  describe this as equal example count, not identical compute.
- The mode-A/B catalog, schema, live answer-check wrapper and API integration
  already support `answer_truncation`.
- PR #84 records quality loss and target/control improvement on Kaggle and Token
  Factory. The claim that these experiments remain unrun is outdated. Both fixes
  remain below their healthy references; neither is a complete agent trace.

The 0.06 minimum effect on Kaggle is a chosen practical decision threshold.
Kaggle uses noise estimated from its own intervention arms. This does not validate
transfer of the Token Factory noise SD (0.019), signature likelihoods or priors.
The Token Factory comparison provides mode-A evidence on one task/model; fresh
multi-task calibration and an end-to-end investigation remain separate work.

## Follow-up execution boundaries

Mode B requires its trainer's explicit context length; the length probe is
unavailable without one. Loading frozen hypotheses or resuming an investigation
cannot bypass the mode-specific catalog. New intervention events record the
statistical settings and skipped permutation gates in `comparison_evidence`;
`verdict-v3` uses those fields. See the [full fault review](fault-evidence-review.md).
