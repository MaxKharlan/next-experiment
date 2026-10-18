"""The README's architecture diagram, as animated SVGs for dark and light pages.

The parts stay in place while one investigation draws its six steps over them, in the
order the README lists them, then fades and starts again. The scene is drawn with
ManimGX and written by svg.py from ManimGX's documentation (MIT, LICENSE.manimgx).
ManimGX needs Python 3.13 or newer, so it runs outside the project's environment:

    uv run --no-project --python 3.14 --with manimgx==0.1.1 python docs/animation/architecture.py
"""

import manimgx as m
import numpy as np

from films import THEMES, text, write

LIFT = 0.1  # the layout sits this much above the frame's center
TIP = 0.17  # length of an arrowhead

# scene units, 16 by 8. The app's parts and the services' share left and right edges.
APP, SERVICES = (-4.0, 0.7), (3.35, 7.45)
HEIGHT = 0.9
PARTS = {  # name: left and right edges, bottom edge, title, the line under it
    "screen": ((-7.75, -5.55), 2.1, "Screen", "in the browser"),
    "api": (APP, 2.1, "API", "FastAPI, server-sent events"),
    "causal": (APP, -2.2, "Causal check", "fix and control, exact-match scoring"),
    "storage": (APP, -3.55, "Storage", "local folder or Nebius Object Storage"),
    "tavily": (SERVICES, 2.1, "Tavily", "known problems, cached"),
    "super": (SERVICES, 0.3, "Nemotron 3 Super", "causes and their probabilities"),
    "ultra": (SERVICES, -0.9, "Nemotron 3 Ultra", "verdict and recipe patch"),
    "jobs": (SERVICES, -2.2, "Post-training API", "LoRA jobs: checks and retrain"),
}
WORKER = (-0.9, 1.6)  # bottom and top of the worker, around its two chips
CHIPS = {  # name: bottom edge, words
    "loop": (-0.05, "Agent loop and model router"),
    "planner": (-0.75, "Planner and level-0 probes"),
}
# each step: its arrows, where its number goes (None: halfway along the first arrow),
# and the parts it lights. The README's list of steps has the same numbers.
STEPS = [
    ([((0.7, 1.35), (3.35, 2.3))], None, ["tavily"]),  # known issues
    ([((0.7, 0.75), (3.35, 0.75))], None, ["super"]),  # causes
    ([], (-4.0, -0.5), ["planner"]),  # free probes
    (  # causal check: the jobs go out and the adapters come back
        [((0.7, -1.6), (3.35, -1.6)), ((3.35, -1.9), (0.7, -1.9))],
        None,
        ["causal", "jobs"],
    ),
    ([((0.7, -0.45), (3.35, -0.45))], None, ["ultra"]),  # verdict
    ([((-5.55, 2.35), (-4.0, 2.35))], None, ["api", "causal", "jobs"]),  # retrain
]


def point(x: float, y: float) -> np.ndarray:
    return np.array([x, y + LIFT, 0.0])


class Architecture(m.Scene):
    palette: dict[str, str] = THEMES["dark"]

    def outline(self, x0, x1, y0, y1, color: str) -> m.RoundedRectangle:
        rect = m.RoundedRectangle(width=x1 - x0, height=y1 - y0, corner_radius=0.12)
        rect.move_to(point((x0 + x1) / 2, (y0 + y1) / 2)).set_stroke(color, width=2)
        return rect.set_fill(self.palette["soft"], 0)

    def lines(self, title: str, sub: str, size=20, weight="SEMIBOLD") -> m.VGroup:
        p = self.palette
        words = [text(title, p["ink"], size, weight)]
        if sub:
            words.append(text(sub, p["muted"]))
        return m.VGroup(*words).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.13)

    def box(self, x0, x1, y0, y1, words: m.VGroup) -> m.RoundedRectangle:
        """A part: its outline, and its words at its left."""
        rect = self.outline(x0, x1, y0, y1, self.palette["strong"])
        words.align_to(rect, m.LEFT).shift(m.RIGHT * 0.25).set_y(rect.get_y())
        self.add(rect, words)
        return rect

    def group(self, x0, x1, y0, y1, label: str) -> None:
        """A box around parts that run together, named at its top left."""
        p = self.palette
        name = text(label, p["muted"], 15, "MEDIUM")
        name.move_to(point(x0 + 0.3, y1 - 0.3), aligned_edge=m.LEFT)
        self.add(self.outline(x0, x1, y0, y1, p["line"]), name)

    def arrow(self, start, end, color: str, width=3.0) -> tuple[m.Line, m.Polygon]:
        a, b = point(*start), point(*end)
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0], 0.0]) * TIP * 0.42
        base = b - d * TIP
        line = m.Line(a, base).set_stroke(color, width=width)
        tip = m.Polygon(b, base + n, base - n).set_stroke(width=0)
        return line, tip.set_fill(color, 1)

    def badge(self, number: int, at) -> m.VGroup:
        p = self.palette
        disc = m.Circle(radius=0.2).set_stroke(width=0).set_fill(p["accent"], 1)
        digit = text(str(number), p["on_accent"], 15, "BOLD")
        return m.VGroup(disc.move_to(point(*at)), digit.move_to(point(*at)))

    def construct(self) -> None:
        p = self.palette
        self.group(-4.3, 1.0, -3.85, 3.6, "App, one Python process")
        self.group(3.05, 7.75, -2.5, 1.85, "Nebius Token Factory")
        parts = {
            name: self.box(x0, x1, y0, y0 + HEIGHT, self.lines(title, sub))
            for name, ((x0, x1), y0, title, sub) in PARTS.items()
        }
        worker = self.outline(*APP, *WORKER, p["strong"])
        title = self.lines("Investigation worker", "a thread for each investigation")
        self.add(worker, title.move_to(point(-3.75, 1.07), aligned_edge=m.LEFT))
        for name, (y0, words) in CHIPS.items():
            chip = self.lines(words, "", 16, "MEDIUM")
            parts[name] = self.box(-3.7, 0.55, y0, y0 + 0.5, chip)
        self.add(*self.arrow((-4.0, 2.75), (-5.55, 2.75), p["strong"], width=2))

        self.wait(0.8)
        drawn, visited, current = [], [], []
        for number, (paths, at, lit) in enumerate(STEPS, start=1):
            arrows = [self.arrow(a, b, p["accent"]) for a, b in paths]
            if at is None:
                (x0, y0), (x1, y1) = paths[0]
                at = ((x0 + x1) / 2, (y0 + y1) / 2)
            dim = [parts[n].animate.set_fill(opacity=0) for n in current]
            for line, tip in arrows:
                self.play(m.Create(line), *dim, run_time=0.55, rate_func=m.smooth)
                self.play(m.FadeIn(tip), run_time=0.12)
                drawn += [line, tip]
                dim = []
            if dim:
                self.play(*dim, run_time=0.3)
            mark = self.badge(number, at)
            light = [
                parts[n].animate.set_stroke(p["accent"], width=3).set_fill(opacity=1)
                for n in lit
            ]
            self.play(m.FadeIn(mark), *light, run_time=0.35)
            drawn.append(mark)
            visited += [n for n in lit if n not in visited]
            current = lit
            self.wait(1.0)
        self.play(
            *[parts[n].animate.set_fill(opacity=0) for n in current], run_time=0.4
        )
        self.wait(3.2)
        self.play(
            *[m.FadeOut(x) for x in drawn],
            *[parts[n].animate.set_stroke(p["strong"], width=2) for n in visited],
            run_time=0.6,
        )
        self.wait(0.4)


if __name__ == "__main__":
    write(Architecture, "architecture")
