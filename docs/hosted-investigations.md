# Hosted investigations

The public interface embeds a separate FastAPI service from `apps/judge`. It runs
the existing Nemotron router, hypothesis/probe loop and Token Factory pair runner.
The Python service and CPU evaluator share one container; GitHub Pages serves the
entry page. This service has its own upload contract and does not expose the
self-hosted API's path-based inputs, code execution or investigation listing.

The deployment uses an existing VPS with a separate Compose project. Its hard
limits are 4.5 GiB memory, no swap, 1.5 CPU and 128 processes. CPU shares are 256.
The API has no Docker socket or access to the other application's volumes. Caddy
and this API share a dedicated proxy network. Training and Nemotron inference
use Token Factory. Adapters are evaluated sequentially on CPU, each in a child process that releases
its model memory before the next starts. The same container limit covers the API
and every child process.

`deploy/judge/Dockerfile` installs a CPU-only PyTorch wheel. The source package
must include `core`, `apps`, `bench/faults` and `jobs/sandbox`: the latter two are
import dependencies, even though the public service never enables sandbox checks.
`python -m apps.judge.preflight` imports the investigation components and measures
model inference with rank-64 LoRA and a long synthetic prompt inside the same
container limits. Run this before exposing a new deployment.

## Configuration

The service needs `TOKEN_FACTORY_API_KEY`, `JUDGE_HF_TOKEN` and
`JUDGE_PRIVATE_REPO`. The private Hugging Face dataset repository must already
contain `wallet.json`, with `version: 1`, `cap_usd: 20.0` and an initially empty
`runs` object. Initialization is a separate administrative operation. The service
refuses public storage and cannot recreate or reset a wallet. The local ledger
reserves the complete $20 allocation before any live run.

The Compose file reads a protected `/opt/next-experiment/env` file. Deployment
transfers credentials directly to that file; they are never part of the image,
repository, browser bundle or report. Runtime data and the model cache use
separate persistent directories. One uvicorn worker is required.

Each request carries a random run identifier and a separate 256-bit access key.
Only a hash of the key is stored. Repeating an identical request returns the same
run; a changed payload with the same identifier is rejected. The private wallet
reserves $1 before any paid operation and uses conditional commits against its
observed repository revision. Concurrent claims fail closed. Per-operation
SQLite reservations remain within this allocation. Unknown outcomes retain their
hold; they are never retried automatically. A completed run releases unused
allowance only when its operation records have all settled. Training estimates
are not provider invoices.

A restart interrupts work and leaves saved evidence and budget holds intact.
Inspect existing provider job IDs and receipts before reconciling an interrupted
run. Do not clear a hold or start replacement training merely to reopen the API.
Do not deploy over a running investigation.

## Scope

Uploads contain four distinct chat JSONL splits. Limits are 16–64 training rows,
4–16 validation rows, 4–8 check and test rows, a 300 KB request, and 4 KB per row.
The model is Qwen3-0.6B, decoding is greedy with 32 output tokens, and the metric
is short-answer exact match. There is one diagnostic pair, with two shuffled
seeds per arm. Small samples use the existing benchmark decision rule and are
explicitly marked exploratory. The supplied recipe and losses are unverified
inputs; an original provider job is not claimed to have been inspected.

The test split stays out of diagnosis and scoring. A full retrain remains an
exported proposal. Generated code, arbitrary filesystem paths and model URLs
cannot be supplied through this endpoint. The browser renders model text as text.
Reports are private and require the access key; there is no public report list.

## Deployment verification

The [VPS preflight](results/hosted-judge-vps-preflight-20261020.json) completed
with the API and an inference child in the same 4.5 GiB cgroup. Peak usage was
3.93 GiB, no OOM events occurred, and the existing application's health checks
passed throughout. Host available memory stayed above 3.06 GiB. This checks one
long synthetic prompt with rank-64 LoRA; it does not guarantee every uploaded
case will complete. Two earlier checks stopped before generation because their
host-swap guard was too sensitive; those outcomes are retained in the receipt.

The [integration smoke](results/hosted-judge-smoke-20261020.json) completed two
Nemotron calls, four Token Factory training jobs and four adapter evaluations.
Its diagnosis was inconclusive. Estimated usage was $0.0401286 from the shared
$15 allocation. This is a service check, not a benchmark result or an invoice.


The [browser integration test](results/hosted-judge-browser-e2e-20261020.json)
completed through the public GitHub Pages entry page in Chrome. The downloaded
example was uploaded through all four file controls. The page showed Nemotron's
hypotheses, training steps, four completed adapter evaluations and the final
report; refreshing reconnected to the same run. The final JSON downloaded from
the page matched the saved server result exactly. Each adapter scored 4/4 on the
check split, so the diagnosis was inconclusive.

The $1 reservation became $0.0409245 in recorded estimated usage, releasing
$0.9590755. The displayed allowance moved from $19.95 before the run to $18.95
while reserved, then $19.91 after completion. Prior run records were unchanged.
A fresh private provider-balance check passed, but its reported balance had not
changed between snapshots. Provider billing for this run is therefore not yet
reconciled; the recorded cost includes a conservative training estimate. The
12 service and balance-guard tests passed, and the neighboring application's
health endpoint remained healthy during CPU evaluation.
