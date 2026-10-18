"""Plot candidate reliability from saved benchmark predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bench.calibration import analyze
from bench.cases import load_cases


def plot_report(report, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted(report["methods"])
    fig, axes = plt.subplots(
        len(methods), 2, figsize=(10, 4 * len(methods)), squeeze=False
    )
    for row, method in enumerate(methods):
        metrics = report["methods"][method]["metrics"]["all"]
        occupied = [b for b in metrics["bins"] if b["count"]]
        ax, counts = axes[row]
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        x = [b["mean_confidence"] for b in occupied]
        y = [b["accuracy"] for b in occupied]
        ax.scatter(x, y, color="#235789", zorder=3, label="Observed bins")
        for b in occupied:
            ax.annotate(
                f"n={b['count']}",
                (b["mean_confidence"], b["accuracy"]),
                xytext=(0, -15 if b["accuracy"] > 0.8 else 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.set(
            xlim=(-0.05, 1.05),
            ylim=(-0.08, 1.08),
            xlabel="Mean reported confidence",
            ylabel="Fraction correct",
            title=f"{method}: n={metrics['n']}, ECE={metrics['ece']:.3f}, "
            f"Brier={metrics['binary_brier']:.3f}",
        )
        ax.legend(loc="lower right", fontsize=8)
        bins = metrics["bins"]
        counts.bar(
            [b["lower"] for b in bins],
            [b["count"] for b in bins],
            width=[b["upper"] - b["lower"] for b in bins],
            align="edge",
            edgecolor="white",
            color="#235789",
        )
        counts.set(
            xlim=(0, 1),
            xlabel="Reported confidence",
            ylabel="Case count",
            title="Bin support (empty bins are not accuracy zero)",
        )
        counts.set_yticks(range(max(b["count"] for b in bins) + 1))
    fig.suptitle(f"Candidate reliability | {report['split']} | descriptive audit")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    try:
        fig.savefig(destination / "reliability.svg")
        fig.savefig(destination / "reliability.png", dpi=160)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory already exists")
    raw = args.input.read_bytes()
    source = raw.decode()
    records = (
        json.loads(source)
        if source.lstrip().startswith("[")
        else [json.loads(line) for line in source.splitlines() if line.strip()]
    )
    cases = load_cases(args.split)
    report = analyze(records, cases, args.bins)
    report.update(
        split=args.split,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        truth_sha256=hashlib.sha256(
            json.dumps({c.id: c.truth for c in cases}, sort_keys=True).encode()
        ).hexdigest(),
    )
    args.out.mkdir(parents=True, exist_ok=False)
    plot_report(report, args.out)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved reliability audit to {args.out}")


if __name__ == "__main__":
    main()
