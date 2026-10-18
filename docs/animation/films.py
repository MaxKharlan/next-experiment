"""What the README's animations share: the screen's colors, the type, and the writer.

Each scene is drawn once per page theme and written by svg.py as an SVG that loops.
The scenes fade their own marks out at the end, so the films start again with no
fade of their own.
"""

from pathlib import Path

import manimgx as m

import svg

OUT = Path(__file__).resolve().parents[1] / "media"
SIZE, FPS = (1280, 640), 30
FONT = "Noto Sans"
# svg.py resamples long outlines for a camera that moves; ours stay, so glyphs such
# as "m" keep all their curves
svg.EXACT = 10_000

# the investigation screen's colors (apps/api/static/index.html)
THEMES = {
    "dark": {
        "ink": "#E4EBE9",
        "muted": "#97A6A2",
        "line": "#2E3D3A",
        "strong": "#62736F",
        "accent": "#5DBF9F",
        "soft": "#16302A",
        "on_accent": "#0E1413",
    },
    "light": {
        "ink": "#152120",
        "muted": "#5A6866",
        "line": "#D6DEDB",
        "strong": "#849390",
        "accent": "#1E6B57",
        "soft": "#E1F0EB",
        "on_accent": "#FFFFFF",
    },
}


def text(words: str, color: str, size: int = 16, weight: str = "NORMAL") -> m.Text:
    return m.Text(words, font=FONT, font_size=size, weight=weight, color=color)


def write(scene: type[m.Scene], name: str) -> None:
    """Write docs/media/<name>-dark.svg and <name>-light.svg; the scene reads
    self.palette."""
    m.config.pixel_width, m.config.pixel_height = SIZE
    m.config.frame_rate = FPS
    for theme, palette in THEMES.items():
        themed = type(
            f"{scene.__name__}{theme.title()}", (scene,), {"palette": palette}
        )
        file = OUT / f"{name}-{theme}.svg"
        svg.write(svg.record(themed), file, fade=0)
        # with fade=0 the root group keeps a fade that never shows; leave it out
        film = file.read_text(encoding="utf-8")
        start = film.index("<g>") + len("<g>")
        end = film.index("/>", start) + len("/>")
        if 'attributeName="opacity"' in film[start:end]:
            file.write_text(film[:start] + film[end:], encoding="utf-8")
        print(file, f"{file.stat().st_size / 1024:.0f} KiB")
