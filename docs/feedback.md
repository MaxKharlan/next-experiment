# Feedback log

Running notes on Nebius Token Factory, Nebius AI Cloud, NVIDIA Nemotron and Tavily for the Devpost feedback. One entry per finding: date, product, what happened, what we expected.

## 2026-10-14 · Nebius AI Cloud · PermissionDenied before the first payment

Creating a bucket and an access key returned `PermissionDenied` in both the console and the CLI. The actual cause was the missing $25 first payment, but the error did not mention billing. After the payment the same commands worked.

## 2026-10-14 · Nebius AI Cloud · two kinds of service account keys

Access keys from `nebius iam v2 access-key` work only for Object Storage. The Nebius CLI and Serverless Jobs need a separate authorized key (`nebius iam auth-public-key generate`). The console does not make this difference obvious, and we first tried the S3 keys with the CLI.

## 2026-10-15 · Token Factory · structured output docs

The "Structured output & JSON" page passes the JSON Schema directly as `json_schema`. The API rejects that with a 422 and expects the OpenAI shape `{"name": ..., "schema": ...}`. With that shape, Nemotron 3 Nano, Super and Ultra all returned schema-valid JSON on the first attempt.

## 2026-10-15 · Token Factory · Nemotron 3 replies

All three Nemotron 3 models work through the OpenAI-compatible API. Plain-text replies can start with blank lines, and a one-word answer still used dozens of tokens, which looks like hidden reasoning counted in the usage. The structured smoke test cost $0.00017 on Nano, $0.00063 on Super and $0.00427 on Ultra per call.
