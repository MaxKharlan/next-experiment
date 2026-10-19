"""Exercise replay cancellation and rendering with the page's actual script."""

import subprocess
from pathlib import Path


def test_static_replay_state_and_raw_evidence():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "node",
            str(root / "tests/browser/static-replay.cjs"),
            str(root / "apps/api/static/index.html"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
