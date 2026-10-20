import json

import pytest

from scripts.export_rest_code_trace import verify_events


def stream(events):
    return "\n\n".join(
        f"id: {e['seq']}\nevent: {e['type']}\ndata: {json.dumps(e['data'])}"
        for e in events
    )


def test_sse_matches_report_with_keepalives():
    events = [
        {"seq": 0, "type": "started", "data": {"id": "inv-example"}},
        {"seq": 1, "type": "finished", "data": {"status": "inconclusive"}},
    ]
    text = stream(events) + "\n\n: ping\n\n"
    assert verify_events(text, {"events": events}) == events


def test_sse_missing_event_is_rejected():
    events = [
        {"seq": 0, "type": "started", "data": {}},
        {"seq": 2, "type": "finished", "data": {}},
    ]
    with pytest.raises(ValueError, match="incomplete"):
        verify_events(stream(events), {"events": events})


def test_sse_changed_outcome_is_rejected():
    events = [
        {"seq": 0, "type": "started", "data": {}},
        {"seq": 1, "type": "finished", "data": {"status": "inconclusive"}},
    ]
    text = stream(events).replace("inconclusive", "confirmed")
    with pytest.raises(ValueError, match="differs"):
        verify_events(text, {"events": events})
