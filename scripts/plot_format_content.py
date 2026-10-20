"""Plot frozen development outcomes; dots are independent training-order repeats."""

import argparse
import json
from pathlib import Path
from statistics import fmean


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())["decision"]
    arms = ("format", "healthy", "cut", "wrong")
    colors = ("#7a8593", "#167754", "#c23b44", "#b27619")
    panels = (
        (
            "format",
            "strict_exact",
            "Direct serialization",
            "24 supplied-field examples",
        ),
        (
            "content",
            "strict_exact",
            "History generation",
            "48 histories · strict JSON exact",
        ),
        (
            "content",
            "choice_exact",
            "Content discrimination",
            "48 histories · 4 candidates · 2 fields",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, (kind, metric, title, subtitle) in zip(axes, panels, strict=True):
        for index, (arm, color) in enumerate(zip(arms, colors, strict=True)):
            values = report["groups"][arm][kind][metric]
            ax.bar(index, fmean(values), color=color, alpha=0.28, width=0.7)
            ax.hlines(fmean(values), index - 0.3, index + 0.3, color=color, linewidth=2)
            ax.scatter(
                [index - 0.14, index, index + 0.14], values, color=color, s=32, zorder=3
            )
        base = report["base"][kind][metric]
        ax.axhline(
            base,
            linestyle="--",
            color="#39424e",
            linewidth=1,
            label=f"Base: {base:.1%}",
        )
        ax.set(
            title=f"{title}\n{subtitle}",
            ylim=(-0.04, 1.07),
            xticks=range(4),
            xticklabels=("Format", "Healthy", "Cut", "Wrong"),
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower left", frameon=False, fontsize=8)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle(
        "Format and content: new development trainings", fontsize=15, x=0.07, ha="left"
    )
    fig.text(
        0.07,
        0.02,
        "Dots: 3 training-order repeats per arm. Bars: means. Development only; format arm is not compute matched.",
        fontsize=9,
        color="#39424e",
    )
    fig.tight_layout(rect=(0.02, 0.07, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
