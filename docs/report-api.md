# Investigation report API

`GET /api/investigations/{id}/report` returns a JSON snapshot of the investigation. Add `?download=true` to receive an attachment named `investigation-report.json`.

For an existing investigation ID stored in `INV_ID`:

```bash
curl --fail-with-body -sS \
  "http://127.0.0.1:8000/api/investigations/${INV_ID}/report?download=true" \
  --output investigation-report.json
```

The version-1 report contains the input case, protocol when recorded, hypotheses, steps, verdict, retrain plan, approval decision, final model comparison, summary and full event log. `last_event_seq` identifies the final event included in this snapshot.

`execution_status` describes the worker lifecycle. `diagnosis_status` describes its conclusion. A confirmed diagnosis can still be waiting for retrain approval. `partial` remains true for queued, running, waiting and failed investigations; it becomes false when execution finishes. A run interrupted by a process restart is reported as failed with an explanation.

Reports come from the API store's persisted events, including retrain and comparison events. Metadata and events are copied under one lock. The route uses the configured store, including Object Storage. Historical logs can lack protocol details or other fields; those remain null or empty. Summary costs are returned as recorded and may omit full retrain costs.

Unknown IDs return 404. In public-demo mode, live reports return 403 and replay reports remain available. Responses use `Cache-Control: no-store`. This endpoint does not start work, answer approval requests or change an investigation.

## Download from the investigation screen

When an API investigation is selected, the screen shows **Download report (JSON)**
under its status. The link downloads the current persisted snapshot, including
partial progress while running, waiting for approval or failed. The adjacent note
identifies partial reports. Selecting another run updates the link to that run.
The static GitHub Pages demo retains its existing saved-demo JSON link.
