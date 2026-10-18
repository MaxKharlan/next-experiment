"""A short video of the public demo for the README: the finished run, then the replay.

Builds the static site (scripts/build_pages.py), serves it on a free local port, plays a
saved investigation in Google Chrome, records Chrome's screencast and encodes an MP4 with
ffmpeg. The video opens on the finished run, because its first frame is the poster
GitHub shows before anyone presses play. The steps arrive at the pace in PACE, slower
than on the site, so that each one can be read.

Needs Google Chrome and ffmpeg on the machine, and Playwright on top of the project:

    uv run --with playwright python -m scripts.record_demo_video --out demo.mp4

GitHub plays a video in the README when it is an uploaded asset: drag the MP4 into an
issue or pull request comment, and put the link it gets on a line of its own.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import http.server
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from scripts.build_pages import build

PACE = {  # seconds before each kind of step; the site waits 1.5 s before every one
    "hypotheses": 2.5,
    "probe": 3.5,
    "intervention": 3.0,
    "verdict": 3.0,
    "retrain_proposal": 2.5,
    "retrain": 2.0,
    "comparison": 2.5,
}
VIEW = {"width": 1280, "height": 720}  # shot at twice this, encoded at 1920 by 1080
FPS, HOOK_S, FADE_S, LEAD_S, TAIL_S = 30, 2.0, 0.3, 1.0, 3.0
BUTTON = "document.getElementById('start').textContent === {!r}"


def slow_down(site: Path, case: str) -> None:
    path = site / "replays" / f"{case}.json"
    events = json.loads(path.read_text(encoding="utf-8"))
    for event in events:
        event["after_s"] = PACE.get(event["data"].get("kind"), event["after_s"])
    path.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def serve(site: Path) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(Quiet, directory=site)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


async def record(url: str, case: str, theme: str, shots: Path) -> tuple[Path, list]:
    """The finished run as one frame, and the replay as (time, frame) pairs."""
    from playwright.async_api import async_playwright

    frames: list[tuple[float, Path]] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="chrome", headless=True)
        context = await browser.new_context(
            viewport=VIEW, device_scale_factor=2, color_scheme=theme
        )
        page = await context.new_page()

        async def play(skip: bool) -> None:
            await page.goto(url, wait_until="networkidle")
            await page.select_option("#demo", case)
            await page.click("#start")
            await page.wait_for_function(BUTTON.format("Skip to the end"))
            if skip:
                await page.click("#start")
            await page.wait_for_function(BUTTON.format("Replay"), timeout=180_000)

        await play(skip=True)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(1200)  # let the entries finish fading in
        hook = shots / "hook.png"
        await page.screenshot(path=hook)

        await page.goto(url, wait_until="networkidle")
        await page.select_option("#demo", case)
        cdp = await context.new_cdp_session(page)

        async def keep(params: dict) -> None:
            shot = shots / f"{len(frames):05d}.jpg"
            shot.write_bytes(base64.b64decode(params["data"]))
            frames.append((params["metadata"]["timestamp"], shot))
            await cdp.send(
                "Page.screencastFrameAck", {"sessionId": params["sessionId"]}
            )

        cdp.on("Page.screencastFrame", lambda p: asyncio.ensure_future(keep(p)))
        shot = {
            "format": "jpeg",
            "quality": 95,
            "maxWidth": 2 * VIEW["width"],
            "maxHeight": 2 * VIEW["height"],
        }
        await cdp.send("Page.startScreencast", shot)
        await page.wait_for_timeout(LEAD_S * 1000)
        await page.click("#start")
        await page.wait_for_function(BUTTON.format("Skip to the end"))
        await page.wait_for_function(BUTTON.format("Replay"), timeout=180_000)
        await page.wait_for_timeout(TAIL_S * 1000)
        end = await page.evaluate("(performance.timeOrigin + performance.now()) / 1000")
        await cdp.send("Page.stopScreencast")
        await browser.close()
    frames.append((end, frames[-1][1]))
    return hook, frames


def encode(hook: Path, frames: list, shots: Path, out: Path) -> None:
    """One output frame every 1/FPS s, each the newest screencast frame at that time."""
    seq = shots / "seq"
    seq.mkdir()
    start, end = frames[0][0], frames[-1][0]
    k = 0
    for i in range(round((end - start) * FPS)):
        while k + 1 < len(frames) and frames[k + 1][0] <= start + i / FPS:
            k += 1
        shutil.copyfile(frames[k][1], seq / f"{i:05d}.jpg")
    scale = "scale=1920:1080:flags=lanczos:in_range=full:out_range=tv,fps=30,settb=AVTB,format=yuv420p"
    graph = (
        f"[0:v]{scale}[a];[1:v]{scale}[b];"
        f"[a][b]xfade=transition=fade:duration={FADE_S}:offset={HOOK_S - FADE_S}[v]"
    )
    subprocess.run(
        [
            *("ffmpeg", "-y", "-loglevel", "error"),
            *("-loop", "1", "-framerate", str(FPS), "-t", str(HOOK_S), "-i", str(hook)),
            *("-framerate", str(FPS), "-i", str(seq / "%05d.jpg")),
            *("-filter_complex", graph, "-map", "[v]"),
            *(
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "slow",
                "-crf",
                "20",
            ),
            *("-tune", "stillimage", "-movflags", "+faststart", str(out)),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("demo.mp4"))
    parser.add_argument(
        "--case", help="a saved investigation in demo/; the first by default"
    )
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is needed to encode the video")
    with tempfile.TemporaryDirectory() as tmp:
        site, shots = Path(tmp, "site"), Path(tmp, "shots")
        shots.mkdir()
        names = build(site)
        case = args.case or names[0]
        if case not in names:
            raise SystemExit(
                f"no saved investigation {case!r}; there are {', '.join(names)}"
            )
        slow_down(site, case)
        server = serve(site)
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/"
            hook, frames = asyncio.run(record(url, case, args.theme, shots))
        finally:
            server.shutdown()
        encode(hook, frames, shots, args.out)
    seconds = HOOK_S - FADE_S + frames[-1][0] - frames[0][0]
    print(f"{args.out}: {seconds:.1f} s, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
