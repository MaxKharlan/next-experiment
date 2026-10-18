# B24: label evidence in mode-B investigations

The Python `investigate` entry point accepts an optional `case["training_labels"]`
attachment in mode B. It checks labels before Tavily or hypothesis generation,
records a `training_label_checks` step through the normal `on_step` callback,
and gives the hypothesis model the aggregate results. The full token and label
arrays are removed from the copied case before model calls; the caller's case
is not modified. Saved investigation reports include the aggregate step.

Example attachment:

```python
case["training_labels"] = {
    "reference": [{"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, 2]}],
    "observed": [{"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, -100]}],
    "eos_token_id": 2,
}
```

The reference must contain healthy, unpadded, unshifted answer-only labels;
ignored reference positions must be prompt positions. The observed side must
contain the same input sequences with the actual labels to inspect. The caller
owns reference correctness and its relationship to the real training job.
There is no automatic extraction from a remote checkpoint or training code.

Validation requires integer token IDs, nonnegative IDs, labels that are token
IDs or `-100`, aligned sequences, and reference labels equal to their token ID
when supervised. Booleans and extra fields are rejected. Each side allows
1–1024 rows, each row at most 32768 positions, with at most 131072 positions
across both sides. Mode A rejects the attachment. Invalid inputs fail before
external calls. An EOS ID is optional; without it, the EOS check is unavailable.

The aggregate contains the existing alignment, prompt-supervision and EOS
masking audits. Masking and prompt-addition audits return `not_applicable` when
the supplied change violates their assumptions. For example, double shifting
must not be interpreted as simple EOS removal. Missing EOS supervision remains
`no_eos_targets`, not evidence of healthy stopping behavior.

These checks cost no provider credits. Their step is added to the evidence list,
but it does not produce a causal verdict, patch or calibrated Bayesian update.
The hypothesis model can use the observations when proposing hypotheses. Frozen
initial hypotheses retain their priors. No new signature probabilities are
invented for these checks. Mode-B intervention builders and downstream quality
measurements remain separate work; this addition does not make an end-to-end
mode-B repair workflow available.

## Verification

```bash
uv run pytest -q tests/unit/test_agent_label_checks.py tests/unit/test_loop.py
```

Tests cover healthy labels, injected faults 9–11, absent EOS IDs, malformed or
oversized attachments, token misalignment and mode gating. Loop tests check
aggregate forwarding, raw-array removal, caller nonmutation, callback delivery,
unchanged frozen input data and absence of automatic causal confirmation. All
inputs are synthetic; no Kaggle job, model download or provider call was used.
The REST API and UI do not yet expose this attachment contract.
