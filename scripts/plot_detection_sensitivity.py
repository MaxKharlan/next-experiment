"""Plot observed pre-intervention top-1 recovery and healthy recipe false alarms."""

import argparse
import json
from pathlib import Path

from bench.sensitivity import wilson_interval


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    keys = sorted({(c["study"], c["fault"]) for c in report["curves"]})
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.8), squeeze=False)
    for ax, (study, fault) in zip(axes.flat, keys, strict=True):
        curves = [
            c for c in report["curves"] if (c["study"], c["fault"]) == (study, fault)
        ]
        baseline = next(h for h in report["healthy"] if h["study"] == study)
        field = curves[0]["parameter"]
        hits = baseline["top1_counts"][fault]
        total = baseline["total"]
        base = baseline["hyperparameters"][field]
        points = [(base, hits / total, wilson_interval(hits, total))] + [
            (c["value"], c["top1_recovery"], c["wilson_95"]) for c in curves
        ]
        points.sort()
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        error = [
            [max(0, p[1] - p[2][0]) for p in points],
            [max(0, p[2][1] - p[1]) for p in points],
        ]
        ax.errorbar(x, y, yerr=error, marker="o", capsize=4, color="#2365a8")
        ax.scatter(
            [base],
            [hits / total],
            marker="s",
            s=65,
            color="#b6482e",
            label="Healthy: same class first",
            zorder=3,
        )
        ax.axhline(
            0.8,
            color="#666666",
            linestyle="--",
            linewidth=1,
            label="80% observed target",
        )
        if field == "learning_rate":
            ax.set_xscale("log")
        ax.set_xticks(x, [f"{v:g}" for v in x])
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel(field)
        ax.set_ylabel("Fraction ranked top-1")
        ax.set_title(
            study.replace("-banking77-500-dev", "") + "\n" + fault.replace("_", " "),
            fontsize=11,
        )
        ax.grid(alpha=0.2)
        ax.legend(
            fontsize=7, loc="lower right" if field == "n_epochs" else "upper right"
        )
    fig.suptitle(
        "Agent ranking before intervention: 3 cases per point, Wilson 95% intervals",
        fontsize=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
