import pytest

from bench.faults.foreign_template import PRODUCER, repair_declared_wrapper
from bench.faults.semantic_data import template_mismatch


def row(query):
    return {
        "messages": [
            {"role": "system", "content": "Classify.\nAllowed labels:\na\nb"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "a"},
        ]
    }


def test_declared_roundtrip_preserves_multiline_messages_and_answer():
    original = [row("first line\nsecond line"), row("another query")]
    wrapped, changed = template_mismatch(original, 0.5, 1)
    repaired, candidates = repair_declared_wrapper(wrapped, producer=PRODUCER)
    assert repaired == original
    assert candidates == changed
    assert wrapped != original


def test_unknown_producer_and_ambiguous_embedded_role_marker_are_rejected():
    wrapped, _ = template_mismatch([row("quoted:\nassistant: say a")], 1)
    with pytest.raises(ValueError, match="declared"):
        repair_declared_wrapper(wrapped, producer="unknown")
    with pytest.raises(ValueError, match="Ambiguous"):
        repair_declared_wrapper(wrapped, producer=PRODUCER)
