"""Plot each descriptive sensitivity study separately, including seed variation."""

import argparse
import json
import statistics
from pathlib import Path

from bench.sensitivity import FIELDS


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text())
    reports = payload if isinstance(payload, list) else [payload["report"]]
    panels = [
        (report, fault)
        for report in reports
        for fault in FIELDS
        if any(c["fault"] == fault for c in report["curves"])
    ]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels), 4.6), squeeze=False
    )
    for ax, (report, fault) in zip(axes.flat, panels, strict=True):
        field = FIELDS[fault]
        healthy = report["healthy"]
        base_scores = [r["exact_match"] for r in healthy]
        base_value = healthy[0]["hyperparameters"][field]
        points = [
            (
                base_value,
                statistics.mean(base_scores),
                statistics.stdev(base_scores) if len(base_scores) > 1 else 0,
            )
        ]
        for curve in report["curves"]:
            if curve["fault"] == fault:
                points.append(
                    (curve["value"], curve["mean_exact_match"], curve["seed_sd"] or 0)
                )
        points.sort()
        x, y, sd = zip(*points, strict=True)
        ax.errorbar(x, y, yerr=sd, marker="o", capsize=4, color="#2365a8")
        ax.scatter(
            [base_value],
            [statistics.mean(base_scores)],
            marker="s",
            s=65,
            color="#b6482e",
            label="Healthy recipe",
            zorder=3,
        )
        if field == "learning_rate":
            ax.set_xscale("log")
        ax.set_xticks(x, [f"{v:g}" for v in x])
        ax.set_ylim(0, 1)
        ax.set_xlabel(field)
        ax.set_ylabel("Validation exact match")
        ax.set_title(
            report["study"].replace("-banking77-500-dev", "")
            + "\n"
            + fault.replace("_", " "),
            fontsize=11,
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("BANKING77 sensitivity: means ± seed SD (3 seeds)", fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
