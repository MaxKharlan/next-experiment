# Resume investigation events

`GET /api/investigations/{id}/events` assigns each SSE message an `id` equal
to its persisted, zero-based event sequence. The browser's `EventSource`
sends the last received ID in `Last-Event-ID` when it reconnects. The API
then sends events after that sequence, including events saved during the
disconnection. This prevents reconnects from replaying steps and adding
training costs to the screen twice.

A new connection without the header replays the full history. Opening another
investigation still starts from the beginning. An empty header also starts
from the beginning. A nonempty header must contain 1–18 ASCII decimal digits;
invalid values return HTTP 422. IDs belong to one investigation and must not
be reused across investigations. A cursor beyond the current history skips
all events up to that sequence.

IDs survive a store restart. Once a finished or failed investigation has no
remaining events, its stream closes. This does not restart an interrupted
worker or provide browser state recovery after a page reload: a reload opens
a new stream and rebuilds the screen from the complete history.

Validation uses saved local events through `TestClient`, reconnecting at
every event boundary and reopening the store. No provider requests or
training jobs are required.
