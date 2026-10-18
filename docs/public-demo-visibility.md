# Public demo visibility

With `PUBLIC_DEMO=true`, the investigation list includes only replay runs.
Requests for a live (`tf` or `local`) investigation return HTTP 403 from the
status, event stream and report endpoints, including report downloads. The
check happens before an SSE response begins. Replay histories and reports
remain readable. Live launches and retraining remain disabled.

This prevents a public demo sharing an existing investigation store from
serving live case data, dataset paths, provider job IDs and saved events.
With `PUBLIC_DEMO=false`, the local API keeps access to all backends.

This is a demo-mode restriction, not user authentication or tenant isolation.
Replay artifacts are public content; review them before deployment. Use a
separate store for the deployed demo where possible. The local API should not
be exposed publicly with live mode enabled.

Tests use saved local runs for all three backends and cover list filtering,
status, SSE, JSON reports and downloads. No provider requests are made.
