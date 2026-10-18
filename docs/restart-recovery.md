# Interrupted investigations after an API restart

On the first read after an API restart, the store appends a persisted `failed`
event to an investigation that was queued or running. The event has
`reason: process_restart` and explains that saved results remain available.
The screen receives the error through SSE and stops waiting. The JSON report
includes the same error and remains marked as partial.

Previous steps, recorded LLM usage and provider job IDs remain in the event
history. Reopening the store does not append another failure. Concurrent reads
within one process share the same recovery operation.

An investigation waiting for approval stays waiting. Finished and failed runs
keep their existing terminal events. Because the store writes events before
metadata, recovery uses the latest saved status transition if metadata is
older. This includes an approval already consumed before the process stopped.

Recovery does not submit or cancel provider jobs. A provider job can continue
after the API process stops; inspect its status before starting another run.
Automatic worker resumption and coordination between multiple API processes
are not implemented. Use one API process for a shared investigation store.

Tests cover queued and running investigations, concurrent reads, repeated
restarts, stale metadata, approval preservation, and the SSE/report output.
They use local storage and make no provider requests.
