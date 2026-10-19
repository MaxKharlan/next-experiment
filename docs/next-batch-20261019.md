# Track B follow-up: B9, B12 and B17

The original package was developed on `618db22` (PR #78) and integrated
with `0b0db5f` (PR #84) on October 19, 2026 in a separate clone. A later local
sweep completed 210 LoRA runs and is analyzed in `b9-final-sweep.md`.

## Selection and completion boundaries

B4/B6/B7 and A2/A3 have since reached main. B5 and the complete B10 matrix
still need reference and calibration runs on those tasks. B8 pair builders and B11 statistics already exist;
B8 intentionally uses a more expensive unchanged-epoch control for fault 7.
B14 sampling is implemented and has a saved dev comparison. The first remaining
items selected here are B9, B12 and B17.

| Item | Delivered and exercised | Still required for full completion |
| --- | --- | --- |
| B9 | 210 simulations, exact-match and loss analysis, grouped classifiers and frozen Token Factory transfer | Transfer fails; keep the classifier offline. A publishable B26 model needs new provider-compatible evidence. |
| B12 | Prior temperature fitting, ECE/Brier reliability plots, leave-one-case-out and eight-case heldout audits | Heldout metrics reject the fitted temperature; keep T=1 until a new independent audit supports a change. |
| B17 | SQLite spending reservations, retained uncertain charges, Python and REST/UI paid-step approval/resume, persisted states and failure snapshots | Provider invoice reconciliation and multiworker ownership remain operational extensions. |

## B9: completed sweep rejects provider transfer

The final study has 210 runs: seven recipes at 30 seeds, with 50 BANKING77
training rows and 16 validation rows. Low learning rate scores zero exact match
on all 60 runs. Rank 2 and rank 4 do not lower quality. Three and six epochs
improve mean exact match over the one-epoch healthy recipe.

The four-family recipe classifier reaches 63.8% balanced accuracy under
leave-one-seed-out evaluation and 59.6% when both seed half and strength change.
A binary classifier for the only observed failure, low learning rate, reaches
96.7% balanced accuracy locally.

Frozen transfer reverses that result. The binary model scores 41.7% balanced
accuracy on five development Token Factory cases and 25.0% on eight historical
heldout cases. It misses all six heldout low-LR cases. See
[`b9-final-sweep.md`](b9-final-sweep.md) and the saved
[`report.json`](results/b9-sweep-50-20261019/report.json).

The classifier remains offline. B9's experiment is complete with a negative
transfer result; B26 publication still lacks a provider-compatible model.

### Earlier transfer baseline

The earlier 15-run local model was frozen before this evaluation; it was neither refitted nor
selected using these predictions. The same 13 cases were inspected historically,
so this is exploratory transfer, not a new untouched test.

| Split | Accuracy | Balanced accuracy over observed classes | Healthy false positives |
| --- | --- | --- | --- |
| Dev | 3/5 | 50% | 2/2 |
| Historical heldout | 6/8 | 50% | 2/2 |

Every prediction is `lr_too_low`. The raw accuracy equals a constant majority-class
prediction. There are no true small-rank or excess-epoch failures in this transfer
set. Per-epoch Token Factory batch losses differ from the local per-step traces.
Do not connect this classifier to live diagnosis or publish it as validated.
The evaluator was recovered from an existing deferred local script, reviewed,
covered by tests and rerun against this clone's versioned cases.

```bash
uv run python -m bench.loss_transfer \
  --model docs/results/loss-classifier-local-20261018/model.json \
  --out artifacts/next-batch/loss-transfer.json
```

## B12: fitting improves dev NLL but hurts cross-validation

Calibration consumes the first saved Super draw on each dev case, before probes
or causal checks. Duplicate fault classes are combined for fitting. Unlisted
probability mass is an explicit category meaning none of the proposed classes;
it is not a probability of health. On a healthy case that category is the target.
A true fault absent from the proposals also targets unlisted mass. Compound truth
is rejected by this single-label protocol.

For temperature T, probabilities are proportional to `max(p, 1e-9) ** (1/T)`.
T=1 is an exact identity. The epsilon permits a zero-mass category to receive mass
under other temperatures. In live use duplicate locations retain their original
within-class shares. No new hypotheses or healthy diagnosis are created.

| Metric | Raw priors | Fitted on all five dev cases | Leave-one-case-out |
| --- | --- | --- | --- |
| NLL | 1.730145 | 1.523515 | 1.795731 |
| Multiclass Brier | 0.811000 | 0.781409 | 0.865918 |

The fitted T is approximately 20, the upper search boundary. Top-category ECE
increases from 0.040 to 0.173 on the fitting data. The plot's Brier label refers
to binary correctness of the top category, while the table above uses the full
distribution. Each cross-validation temperature excludes its evaluated case;
these five related cases still cannot establish generalization.

Keep `prior_temperature=1.0`, the unchanged default. Python callers can explicitly
pass a positive `prior_temperature` to `investigate` or `investigate_with_answers`;
the report records before/after distributions. This experimental option is not
silently enabled in the API. Saved calibration results are in
[the audit](results/prior-calibration-20261019.json) and
[the reliability plot](results/prior-calibration-20261019/reliability.png).

```bash
uv run --with matplotlib python -m bench.prior_calibration \
  --draws docs/results/dev-hypothesis-sampling-20261018 \
  --out artifacts/next-batch/prior-calibration.json --plot
```

## B12: heldout audit rejects deployment

Eight first-draw Super predictions were collected on the historical heldout
cases. The temperature fitted on development cases was applied without refitting.
Heldout NLL worsened from 1.556177 to 1.600629, multiclass Brier from 0.723750
to 0.794069, and ECE from 0.256250 to 0.416064. Keep the default T=1.
These related historical cases do not establish transfer to other tasks or models.

The [heldout audit](results/prior-calibration-heldout-20261019.json) and
[collection ledger](results/heldout-priors-20261019/summary.json) preserve the
inputs, hashes and measured usage. The reliability PNG in the heldout result
folder still plots the development fitting rows; use the JSON `heldout` metrics
for the heldout conclusion.

## B17: reserve spending before execution

`SpendingLedger` uses SQLite `BEGIN IMMEDIATE` to serialize reservations across
connections/processes. An operation ID cannot be submitted twice, and reopening
an existing ledger with a different cap fails. Completed calls settle against
reported token usage. Unknown outcomes and missing usage retain their pending
reservation. Reconciliation is explicit: `settle(id, verified_cost)`; zero is
appropriate only when there is evidence that nothing was charged. Overruns are
recorded truthfully and block further reservations instead of being clipped.

Live API investigations use `artifacts/investigations/<id>/spending.sqlite3`.
Inference, including structured-output retries, reserves before sending. Guarded
requests disable the OpenAI client's hidden retries. Unknown model prices are
rejected. The input estimate uses serialized UTF-8 bytes plus 4096 framing units;
output uses `max_tokens`. This estimate and the configured price table are not a
provider-enforced spending cap. Investigations retain the ledger in successful
summary events and failures. The ledger survives an API restart. Investigations waiting for a paid-step
approval retain their continuation; an interrupted running investigation follows
the existing fail-on-restart policy.

Probes reserve their estimated cost and settle their reported cost. Causal pairs
reserve the recipe estimate and settle that estimate after execution; it is not
an invoice. A timeout leaves the reservation pending. Python callers may supply
`approve_paid_step(proposal)`; only literal `True` allows the paid probe or pair,
and decisions are recorded. Without a callback the existing caller-authorized
budget workflow remains. Live REST/UI investigations pause before each paid
probe or causal pair. `POST /api/investigations/{id}/paid-step` accepts an
approval or refusal once; approval resumes from the persisted continuation
without repeating the initial hypothesis calls. This gate does not pause before
every inference call; inference uses the authorized investigation budget.
Full retraining retains its existing separate approval workflow and is outside
this investigation ledger. Tavily retains its existing credit limits.

Concurrent ledger access does not make the API investigation store safe for
multiple workers. A global account-wide cap, durable worker ownership and
provider billing reconciliation remain separate requirements.

## Validation and spend

Unit coverage checks frozen transfer, duplicate jobs, invalid scalers, probability
mass, duplicate locations, excluded-case fitting, simultaneous reservations,
restart persistence, cap changes, overruns, inference timeouts, refusal before
sending, declined interventions and retained training reservations.
The API's existing end-to-end restart test covers the live integration.

The saved heldout collection ledger reports eight Super calls costing
$0.0135807 against its $0.50 cap, with all eight operations settled. This is
usage-based accounting, not a provider invoice. The local sweep has no provider
training charge. Integration and testing made no new provider calls.
Local audits do not establish a price for future provider simulations.
