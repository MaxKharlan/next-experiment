# Testing Next Experiment

[Open Next Experiment](https://maxkharlan.github.io/next-experiment/).

$20 is allocated to additional live investigations. Testing is free and requires no
API key. Choose **Run with Nemotron**, or open the
[hosted investigation form](https://next.46-17-103-231.sslip.io/) directly. It shows
the remaining budget before starting. Recorded
investigations and local label checks do not spend it.

## Run an investigation

Describe a failed fine-tuning run, enter its recipe, and upload four chat JSONL
splits: 16–64 training rows, 4–16 validation rows, 4–8 check rows and 4–8 test rows.
Every row contains a `messages` array ending in an assistant answer. Use distinct
records across splits. This hosted runner supports Qwen3-0.6B and short answers
scored by exact match, with at most 32 generated tokens.

Nemotron proposes hypotheses and interprets probes. The investigation loop can
submit a targeted fix and its control to Token Factory, with two shuffled seeds
per arm, then evaluate the downloaded adapters on CPU. The selected comparison,
per-example answers and model-call receipts appear in the run evidence. A confirmed
fix includes a recipe proposal; full retraining is a separate operation.

The supplied recipe and loss history are not verified against an original provider
job. Small uploaded samples and inherited benchmark thresholds make this an
exploratory investigation. An inconclusive or failed run retains its partial
evidence. There are no automatic paid retries.

The shared $20 allocation persists across hosting restarts. Each active run
reserves $1 and can execute one diagnostic pair. Completed, accounted operations
release unused allowance; ambiguous failures retain their reservation. One run
executes at a time. Training may wait for provider capacity. Keep the tab open and
download its evidence when finished. The browser stores a private access key to
reconnect after refreshing.

Uploads are processed by the service and sent to Nebius for training. Probe
summaries can send data excerpts to Nemotron. Reports are kept privately on Hugging Face;
there is no public listing of investigations. Do not submit secrets or personal
data.

## Explore without spending the live budget

Replay streams saved investigations and exports their reports. The editable
training-label examples and local JSON upload run entirely in the browser. They
check prompt loss, EOS masking and label alignment. They do not call Nemotron or
train a model.
