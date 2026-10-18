"""The README's causal check: how the demo case proves its cause, as animated SVGs.

The fix and a control of the same cost each train two Post-training jobs. The bars
grow to each job's exact match, the gap between the arms is the effect, and the
effect with its 95% interval is set against the minimum effect. The numbers are the
demo run's (demo/fault6-tf.json). Written like architecture.py:

    uv run --no-project --python 3.14 --with manimgx==0.1.1 python docs/animation/causal_check.py
"""

import manimgx as m
import numpy as np

from films import THEMES, text, write

CONTROL, FIX = (0.12, 0.12), (0.50, 0.50)  # exact match of each seed's job
EFFECT, LOW, HIGH = 0.38, 0.351, 0.409  # effect of the fix and its 95% interval
MIN_EFFECT, NOISE_SD = 0.03, 0.015

BASE, PER_UNIT = -2.7, 8.0  # bar chart: where 0 sits, scene units per 1.0
BARS = {"control": (-5.6, -4.9), "fix": (-2.9, -2.2)}  # x of each seed's bar
AXIS_Y, X0, PER_EFFECT = -0.4, 1.3, 12.0  # effect scale: its line, where 0 sits
GAP_X = -1.5  # the bracket between the arms


def bar_y(value: float) -> float:
    return BASE + value * PER_UNIT


def effect_x(value: float) -> float:
    return X0 + value * PER_EFFECT


# words that stay on screen: text, color, size, weight, x, y, which side sits at x
LABELS = [
    ("The causal check of the demo case", "ink", 22, "SEMIBOLD", -7.6, 3.45, "left"),
    (
        "4 Post-training jobs on Token Factory, two seeds for each arm, $0.055",
        *("muted", 15, "NORMAL", -7.6, 2.95, "left"),
    ),
    ("Exact match", "muted", 14, "MEDIUM", -7.6, 2.2, "left"),
    *[
        (f"{t:.1f}", "muted", 13, "NORMAL", -6.85, bar_y(t), "right")
        for t in (0.1, 0.2, 0.3, 0.4, 0.5)
    ],
    ("0", "muted", 13, "NORMAL", -6.85, BASE, "right"),
    ("Control", "ink", 16, "SEMIBOLD", -5.25, BASE - 0.4, "center"),
    ("the failed recipe again", "muted", 13, "NORMAL", -5.25, BASE - 0.78, "center"),
    ("Fix", "ink", 16, "SEMIBOLD", -2.55, BASE - 0.4, "center"),
    (
        "learning rate 1e-4, not 1e-5",
        "muted",
        13,
        "NORMAL",
        -2.55,
        BASE - 0.78,
        "center",
    ),
    ("Effect of the fix over the control", "muted", 14, "MEDIUM", X0, 2.2, "left"),
    *[
        (
            f"{t / 10:.1f}" if t else "0",
            "muted",
            13,
            "NORMAL",
            effect_x(t / 10),
            AXIS_Y - 0.35,
            "center",
        )
        for t in range(6)
    ],
    (
        f"minimum effect {MIN_EFFECT}",
        "muted",
        13,
        "NORMAL",
        effect_x(MIN_EFFECT) + 0.15,
        AXIS_Y + 1.55,
        "left",
    ),
    (
        f"The interval's width comes from the noise between seeds, SD {NOISE_SD}.",
        *("muted", 13, "NORMAL", X0, AXIS_Y - 0.95, "left"),
    ),
]
SIDES = {"left": m.LEFT, "right": m.RIGHT, "center": None}


class CausalCheck(m.Scene):
    palette: dict[str, str] = THEMES["dark"]

    def label(self, words, color, size, weight, x, y, side) -> m.Text:
        mob = text(words, self.palette[color], size, weight)
        edge = SIDES[side]
        point = np.array([x, y, 0.0])
        return (
            mob.move_to(point, aligned_edge=edge)
            if edge is not None
            else mob.move_to(point)
        )

    def rule(self, x0, y0, x1, y1, color: str, width: float) -> m.Line:
        line = m.Line(np.array([x0, y0, 0.0]), np.array([x1, y1, 0.0]))
        return line.set_stroke(self.palette[color], width=width)

    def construct(self) -> None:
        p = self.palette
        for t in (0.1, 0.2, 0.3, 0.4, 0.5):
            self.add(self.rule(-6.6, bar_y(t), -0.8, bar_y(t), "line", 1.5))
        self.add(self.rule(-6.6, BASE, -0.8, BASE, "strong", 2))
        zone = m.Rectangle(width=effect_x(MIN_EFFECT) - X0, height=1.7)
        zone.move_to(np.array([(X0 + effect_x(MIN_EFFECT)) / 2, AXIS_Y + 0.85, 0.0]))
        self.add(zone.set_stroke(width=0).set_fill(p["line"], 1))
        self.add(self.rule(X0, AXIS_Y, effect_x(0.5), AXIS_Y, "strong", 2))
        for t in range(6):
            x = effect_x(t / 10)
            self.add(self.rule(x, AXIS_Y, x, AXIS_Y - 0.1, "strong", 2))
        for row in LABELS:
            self.add(self.label(*row))

        self.wait(0.6)
        drawn = []
        for arm, values, color in (
            ("control", CONTROL, "strong"),
            ("fix", FIX, "accent"),
        ):
            bars = [
                self.rule(x, BASE, x, bar_y(v), color, 55)
                for x, v in zip(BARS[arm], values, strict=True)
            ]
            self.play(m.LaggedStart(*map(m.Create, bars), lag_ratio=0.25), run_time=0.9)
            top = bar_y(max(values)) + 0.3
            score = self.label(
                f"{values[0]:.2f}",
                "ink",
                15,
                "MEDIUM",
                sum(BARS[arm]) / 2,
                top,
                "center",
            )
            self.play(m.FadeIn(score), run_time=0.25)
            drawn += [*bars, score]
            self.wait(0.3)

        # the gap between the arms is the effect
        low, high = bar_y(CONTROL[0]), bar_y(FIX[0])
        level = self.rule(-4.55, low, GAP_X, low, "strong", 1.5)  # the control's level
        bracket = [
            self.rule(GAP_X, low, GAP_X, high, "accent", 3),
            self.rule(GAP_X - 0.12, low, GAP_X + 0.12, low, "accent", 3),
            self.rule(GAP_X - 0.12, high, GAP_X + 0.12, high, "accent", 3),
        ]
        self.play(m.Create(level), *map(m.Create, bracket), run_time=0.6)
        gain = self.label(
            f"+{EFFECT:.2f}",
            "accent",
            18,
            "SEMIBOLD",
            GAP_X + 0.25,
            (low + high) / 2,
            "left",
        )
        self.play(m.FadeIn(gain), run_time=0.3)
        drawn += [level, *bracket, gain]
        self.wait(0.4)

        # the same effect on its scale, with the interval
        y = AXIS_Y + 0.7
        interval = self.rule(effect_x(LOW), y, effect_x(HIGH), y, "accent", 14)
        dot = m.Circle(radius=0.11).set_stroke(width=0).set_fill(p["accent"], 1)
        dot.move_to(np.array([effect_x(EFFECT), y, 0.0]))
        words = [
            self.label(
                f"+{EFFECT:.2f}",
                "accent",
                18,
                "SEMIBOLD",
                effect_x(EFFECT),
                y + 0.85,
                "center",
            ),
            self.label(
                f"95% interval {LOW:.2f} to {HIGH:.2f}",
                "muted",
                13,
                "NORMAL",
                effect_x(EFFECT),
                y + 0.45,
                "center",
            ),
        ]
        self.play(m.Create(interval), run_time=0.6)
        self.play(m.FadeIn(dot), *map(m.FadeIn, words), run_time=0.4)
        drawn += [interval, dot, *words]
        self.wait(0.4)

        chip = m.RoundedRectangle(width=1.9, height=0.5, corner_radius=0.25)
        chip.move_to(np.array([X0 + 0.95, -2.45, 0.0])).set_stroke(width=0).set_fill(
            p["accent"], 1
        )
        verdict = [
            chip,
            text("Confirmed", p["on_accent"], 16, "BOLD").move_to(chip.get_center()),
            self.label(
                "the fix beats the control beyond the noise",
                "ink",
                15,
                "NORMAL",
                X0 + 2.15,
                -2.45,
                "left",
            ),
        ]
        self.play(*map(m.FadeIn, verdict), run_time=0.4)
        drawn += verdict

        self.wait(3.5)
        self.play(*map(m.FadeOut, drawn), run_time=0.6)
        self.wait(0.4)


if __name__ == "__main__":
    write(CausalCheck, "causal-check")
