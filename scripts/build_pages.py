"""The public demo as a static site for GitHub Pages: the investigation screen and the
saved investigations it replays, with no server, keys or GPU.

    uv run python -m scripts.build_pages --out _site
    python -m http.server -d _site

The screen is apps/api/static/index.html switched to static mode. It lists the cases
from demo-cases.json and plays replays/<name>.json in the browser: the events the
server replay streams, each with the pause before it. demo/<name>.json is the saved
report itself, linked from the page. .github/workflows/pages.yml builds and publishes
the site on pushes to main.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from apps.api import runners

SCREEN = Path(__file__).resolve().parents[1] / "apps" / "api" / "static" / "index.html"
LIVE, STATIC = 'data-mode="live"', 'data-mode="static"'


def build(out: Path) -> list[str]:
    """Write the site to out and return the names of the replayed cases."""
    html = SCREEN.read_text(encoding="utf-8")
    if html.count(LIVE) != 1:
        raise ValueError(f"{SCREEN} must carry {LIVE} once, on <html>")
    names = list(runners.demo_cases())
    if not names:
        raise ValueError(f"no saved investigations in {runners.DEMO_DIR}")
    (out / "replays").mkdir(parents=True, exist_ok=True)
    (out / "demo").mkdir(exist_ok=True)
    (out / "index.html").write_text(
        html.replace(LIVE, STATIC).replace(
            "<script>", '<script src="label-audit.js"></script>\n<script>', 1
        ),
        encoding="utf-8",
    )
    shutil.copyfile(SCREEN.parent / "label-audit.js", out / "label-audit.js")
    shutil.copyfile(SCREEN.parent / "label-examples.json", out / "label-examples.json")
    _write_json(out / "demo-cases.json", runners.demo_index())
    for name in names:
        events = [
            {"after_s": pause, "type": kind, "data": data}
            for pause, kind, data in runners.replay_events(name)
        ]
        _write_json(out / "replays" / f"{name}.json", events)
        saved = runners.DEMO_DIR / f"{name}.json"
        shutil.copyfile(saved, out / "demo" / saved.name)
    return names


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("_site"))
    args = parser.parse_args()
    names = build(args.out)
    print(f"wrote {args.out}: the screen and the replays of {', '.join(names)}")


if __name__ == "__main__":
    main()
