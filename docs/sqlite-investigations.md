# SQLite investigation state (B22)

Set `INVESTIGATION_BACKEND=sqlite` to persist metadata and event history in
SQLite. `DATABASE_URL` selects the file; its default is
`sqlite:///./artifacts/next_experiment.db`. The API saves both JSON columns in
one transaction in `investigation_snapshots`, keyed by investigation ID.

```bash
INVESTIGATION_BACKEND=sqlite \
DATABASE_URL=sqlite:///./artifacts/next_experiment.db \
uv run uvicorn apps.api.main:app --workers 1
```

`STORAGE_BACKEND` still controls artifact storage, including model files and
LLM call logs. `INVESTIGATION_BACKEND=storage` is the default and retains the
existing file/S3 state backend. Switching backends selects a separate history;
existing investigations are not imported automatically. Finish active runs before
switching, and select the previous backend to access its saved investigations.

SQLite state uses the existing API routes, SSE sequence numbers, report builder,
request identities and restart recovery. A waiting proposal survives a restart.
An interrupted queued or running investigation is marked failed when loaded;
provider jobs are not cancelled or restarted. No API keys are stored by this
repository, but case data and events can contain private input; protect the file
as you would the artifacts directory.

Run one API process. Investigation caches, request deduplication and approval
locks remain process-local. This change does not provide distributed workers or
cross-process execution claims. Snapshots rewrite the full event history on each
save, so write cost grows with investigation length.

The API test suite runs against both state backends. Additional tests cover a
failed snapshot write preserving the previous state, fresh-connection recovery,
request identity reuse and one-time approval within the active process.
