# Paid-operation recovery — 19 October 2026

API recipe retraining now reserves its approved estimate in the same per-investigation
SQLite ledger used by investigation steps and answer-drop retraining. Previously,
changing hyperparameters could start training without that reservation. Completion
adds the estimate and ledger snapshot to the investigation summary. The recorded
amount is a training estimate, not a reconciled provider invoice.

Before uploading data, the recipe wrapper checks the plan against the frozen model,
recipe, cost estimate and four dataset hashes. It also checks the evaluation runtime
and paths, then checks the hashes again before scoring. The saved binding prevents
reuse of jobs from another plan. Legacy API proposals without a frozen protocol fail
before constructing a provider client; they need a new proposal with a budget.
Declining a legacy proposal still works.

The retrainer saves each uploaded file ID and a submission marker before creating a
job. If the response is lost, a subsequent call stops for reconciliation instead of
submitting another job. A verified result is saved before budget settlement, allowing
a restart after a settlement failure to return the same result without training or
scoring again. An incomplete recipe attempt remains blocked pending reconciliation,
even when it has known job IDs. The standalone retrain command gains the submission
marker but does not gain the API's frozen-protocol or shared-budget wrapper.

## Tavily credits

Tavily now reserves credits in SQLite before each uncached HTTP request. Admission
checks both the investigation limit and the total limit in one transaction. Multiple
clients using the same ledger cannot spend the last credit twice. Restarting a client
preserves usage and unresolved reservations. A cache hit remains free.

With local storage, the default ledger is `<storage root>/tavily-credits.sqlite3`.
`TAVILY_LEDGER_PATH` overrides it and is required with S3 storage. All callers sharing
a budget must use the same SQLite file on a filesystem that supports SQLite locking.
Separate local files on different hosts do not enforce a shared account limit.

The connector reserves one credit for basic search and two for advanced search.
Extract reserves for all requested URLs, in groups of five, and settles using returned
usage. Missing usage falls back to the estimate; zero usage remains zero. The rates
are documented in [Tavily's credit reference](https://docs.tavily.com/documentation/api-credits).
Automatic search-depth selection is rejected because it could change the reserved
price. Reported overruns are recorded truthfully and reduce subsequent headroom.
Credits remain separate from the investigation's dollar ledger.

Timeouts, HTTP failures, invalid usage and cache-write failures do not permit automatic
resubmission of the same request. The ledger snapshot exposes request IDs, pending
holds and settled charges. The `spent` field is still a session delta for individual
search reports; `total_credits` includes durable settled charges and pending holds.

## Operator recovery and remaining scope

For an interrupted operation, preserve the ledger and saved state, inspect the
provider's job/request evidence, and establish whether the request was accepted before
repairing state. Do not delete a reservation or submission marker merely to retry.
There is no automatic reconciliation command in this change. A saved result can be
reused only with its original plan, protocol and reservation.

The Tavily ledger starts accounting when this version is deployed. Existing cached
responses carry no historical charges into it. Limits do not reset at a calendar-month
boundary, and changing a persisted limit is rejected. Reconcile previous usage and
pending requests before establishing a new budget period. This change does not add
provider invoice import, account-wide billing reconciliation, or ownership leases for
multiple API workers.

Tests use fake providers and temporary storage. They cover API approval and legacy
rejection, changed datasets, insufficient budgets, ambiguous submissions, recovery
after settlement, persistent Tavily limits, concurrent reservations, zero and invalid
usage, overruns, and failed cache writes. No paid requests are needed to run them.
