# Provider cost reconciliation — 19 October 2026

Token Factory Usage confirms $8.0379272 of aggregate fine-tuning charges before
tax. Its four product quantities exactly match the trained-token counts of all
224 successful jobs returned by the provider API. The six completed trainings
in this continuation's ledger match their job-token estimates, totaling
$0.2710464. This establishes agreement with aggregate provider usage; individual
job invoices are unavailable.

## Scope and evidence

The authenticated console showed one project, `default-project`. Usage was read
with full numbers, all ten products in `eu-north1`, for October 1–31, 2026. The
console's last-update marker was October 19 at 12:22 UTC. An initial September
19–October 19 view had identical quantities and amounts. Transactions was then
compared with the October Usage view so the displayed date ranges agree.

The paginated `fine_tuning.jobs.list` GET snapshot at 13:35:18 UTC contains
240 distinct jobs: 224 succeeded, 12 running, one failed and three cancelled.
Every successful job used LoRA with context length 8192. The four failed or
cancelled jobs reported zero tokens; this alone does not prove they incurred no
charge. Their individual billing remains unknown.

| Training product | Successful jobs | API tokens = Usage tokens | Usage charge, USD |
| --- | ---: | ---: | ---: |
| Llama 3.2 1B LoRA 8K | 5 | 206,450 | 0.0825800 |
| Llama 3.1 8B LoRA 8K | 5 | 279,088 | 0.1116352 |
| Qwen3 0.6B LoRA 8K | 209 | 19,448,386 | 7.7793544 |
| Qwen3 1.7B LoRA 8K | 5 | 160,894 | 0.0643576 |
| Total | 224 | 20,094,818 | 8.0379272 |

All four rows agree with $0.40 per million trained tokens. The account-wide
training total includes earlier project experiments and must not be charged
again to the current continuation's $5 ledger.

The six inference product rows total $0.65831022: Ultra $0.235176,
Nano $0.00028212 and Super $0.4228521. Combined with training, Usage totals
$8.69623742 before tax. These are observed provider product totals. Existing
local call logs do not contain complete provider billing references, including
for lost responses, so this audit does not match every inference charge to an
individual local call. Tavily is a separate provider and is outside this snapshot.

## Current continuation

The read-only SQLite snapshot covers six settled historical trainings and 12
pending format/content trainings. All 18 operation-to-job bindings exist in the
provider inventory; no job is bound to two operations. Each historical damaged
run records 111,610 tokens ($0.044644); each healthy run records 114,262 tokens
($0.0457048). Their costs agree within $10^-12, the explicit tolerance for the
existing SQLite REAL representation. Provider token and product-cost comparisons
use decimal arithmetic without that tolerance.

The 12 new jobs were submitted after the Usage cutoff and were running in the
inventory snapshot. Their $1.1311488 reservation remains in place. The ledger's
settled amount plus pending reservations is $1.4021952 against its $5 cap;
reservations are not reported as incurred charges. No reconciliation step writes
the ledger, releases a hold, resubmits a job or increases a budget.

## What remains unresolved

The October Usage header displays subtotal $8.70, VAT $0.78 and total $9.47.
October Transactions displays consumption of $9.48 including tax. The one-cent
difference is recorded; the cause cannot be established from rounded UI values.
It is not corrected by inventing a tax rate or allocating the difference to a
job. The two pages can also have different refresh times.

Payments displayed “No documents”. Downloading the consumption-breakdown PDF
failed with `ERR_SOCKET_NOT_CONNECTED`; its contents were not read. Thus
`invoice_reconciled` remains false. The provider's
[billing documentation](https://docs.tokenfactory.nebius.com/other-capabilities/billing-new)
describes consumption and payment documents separately. A later invoice or
successfully downloaded breakdown is needed to resolve the tax/total difference;
per-call inference matching also needs provider evidence. The new 12-job cohort
must be checked against a Usage snapshot updated after those jobs finish.

## Reproduce the saved comparison

The [input](results/provider-cost-reconciliation-20261019/input.json) contains
manually transcribed product rows, minimal job receipts with hashed IDs, and a
read-only local ledger snapshot. The
[report](results/provider-cost-reconciliation-20261019/report.json) includes its
input SHA-256 and exact totals. API credentials, tenant/contract IDs, training file
IDs, account credits and signed download URLs are omitted. Hashes identify the
private source captures retained locally under ignored `artifacts/`; a public
reader can reproduce the calculations but cannot authenticate the console
transcription from those hashes alone.

```bash
uv run python -m scripts.reconcile_provider_costs \
  --input docs/results/provider-cost-reconciliation-20261019/input.json \
  --out docs/results/provider-cost-reconciliation-20261019/report.json --verify
```

Without `--verify`, the command writes a report from the supplied input. Exit 0
means the calculations agree and no job/token/local-cost discrepancy was found;
it does not mean invoice reconciliation is complete. The output separately
reports the displayed-total difference and invoice status. For a new snapshot,
retain complete API pagination, the selected project and period, the Usage
update time and explicit operation-to-job bindings. `snapshot_ledger` opens SQLite
with `mode=ro`; it never creates a missing database or uses the ledger's write
transaction helper. Failed and unknown outcomes require separate reconciliation
before any paid retry.

Validation: 28 focused checks cover duplicate jobs, duplicate bindings, one-token
differences, missing bindings, invalid amounts/currency, incomplete pagination,
cutoff boundaries, ambiguous failures and a byte-unchanged SQLite snapshot.
The full suite passed: 889 tests, eight skipped. This audit incurred no new
provider charges.
