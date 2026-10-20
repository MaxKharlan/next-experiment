import json

from apps.api import runners
from scripts import build_pages


def test_static_site_plays_what_the_server_replay_streams(tmp_path):
    assert build_pages.build(tmp_path) == list(runners.demo_cases())
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'data-mode="static"' in html and 'data-mode="live"' not in html
    assert '<script src="label-audit.js"></script>' in html
    assert (tmp_path / "label-audit.js").read_bytes() == (
        build_pages.SCREEN.parent / "label-audit.js"
    ).read_bytes()
    assert (tmp_path / "label-examples.json").read_bytes() == (
        build_pages.SCREEN.parent / "label-examples.json"
    ).read_bytes()
    listed = json.loads((tmp_path / "demo-cases.json").read_text(encoding="utf-8"))
    assert listed == runners.demo_index()

    events = json.loads((tmp_path / "replays" / "fault6-tf.json").read_text())
    played = [(e["after_s"], e["type"], e["data"]) for e in events]
    assert played == runners.replay_events("fault6-tf")
    assert played[1][:2] == (runners.REPLAY_DELAY_S, "step")
    saved = (tmp_path / "demo" / "fault6-tf.json").read_bytes()
    assert saved == (runners.DEMO_DIR / "fault6-tf.json").read_bytes()
