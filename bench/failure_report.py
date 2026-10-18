"""Build a traceable failure inventory from existing benchmark score rows."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from bench.cases import load_cases
from bench.score import Row


def classify(row):
    """Categories overlap; they must not be summed into an overall failure rate."""
    categories = []
    if row.truth and row.top1 is False:
        categories.append("wrong_top_candidate")
    if row.truth and row.status != "confirmed":
        categories.append("cause_not_confirmed")
    if row.false_alarm:
        categories.append("false_alarm_on_healthy")
    if row.status == "confirmed" and row.truth and row.cause not in row.truth:
        categories.append("wrong_primary_confirmation")
    if row.wrong_confirmations:
        categories.append("wrong_confirmations")
    if (
        row.status == "confirmed"
        and row.fix_success is False
        and row.fixed_metric is not None
        and row.reference is not None
    ):
        categories.append("measured_fix_failed")
    return categories


def audit(records, cases):
    known = {case.id: case for case in cases}
    seen = set()
    methods = {}
    findings = []
    for number, data in enumerate(records, 1):
        row = Row(**data)
        if row.case not in known or set(row.truth) != set(known[row.case].truth):
            raise ValueError(f"Unknown case or mismatched truth: {row.case}")
        if (row.method, row.case) in seen:
            raise ValueError(f"Duplicate row: {row.method}/{row.case}")
        seen.add((row.method, row.case))
        if row.status not in {"confirmed", "inconclusive"}:
            raise ValueError("Unsupported diagnosis status")
        if row.status == "confirmed" and not row.cause:
            raise ValueError("Confirmed diagnosis must identify a cause")
        if row.status != "confirmed" and row.cause is not None:
            raise ValueError("Inconclusive diagnosis cannot identify a confirmed cause")
        expected_alarm = row.status == "confirmed" if not row.truth else None
        expected_top = bool(row.top) and row.top[0] in row.truth if row.truth else None
        if row.false_alarm is not expected_alarm or row.top1 is not expected_top:
            raise ValueError("Score flags disagree with truth and diagnosis")
        for value in (row.cost_usd, row.p_true, row.fixed_metric, row.reference):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Scores must be finite and nonnegative")
        for value in (row.wrong_confirmations, row.mini_jobs, row.llm_tokens):
            if type(value) is not int or value < 0:
                raise ValueError("Counts must be nonnegative integers")
        categories = classify(row)
        methods.setdefault(row.method, []).append((row, categories))
        if categories:
            findings.append(
                {"source_record": number, "categories": categories, "row": asdict(row)}
            )
    if not methods:
        raise ValueError("No score rows")
    summary = {}
    for method, entries in methods.items():
        if {row.case for row, _ in entries} != set(known):
            raise ValueError(f"Incomplete split for method {method}")
        counts = Counter(category for _, cats in entries for category in cats)
        summary[method] = {
            "cases": len(entries),
            "fault_cases": sum(bool(r.truth) for r, _ in entries),
            "healthy_cases": sum(not r.truth for r, _ in entries),
            "cases_with_findings": sum(bool(cats) for _, cats in entries),
            "category_case_counts": dict(counts),
            "wrong_confirmation_count": sum(r.wrong_confirmations for r, _ in entries),
            "confirmed_without_measured_fix": sum(
                r.status == "confirmed"
                and (
                    r.fixed_metric is None
                    or r.reference is None
                    or r.fix_success is None
                )
                for r, _ in entries
            ),
        }
    return {
        "version": 1,
        "methods": summary,
        "findings": findings,
        "scope": "saved benchmark scores; overlapping categories; no new diagnoses",
    }


def markdown(report):
    def safe(value):
        return (
            html.escape(str(value))
            .replace("|", "&#124;")
            .replace("\n", " ")
            .replace("\r", " ")
        )

    lines = [
        "# Saved benchmark failure inventory",
        "",
        "Generated from saved score rows. Categories overlap; their counts are not additive.",
        "Missing fix measurements are reported separately, not counted as failed fixes.",
        "",
        "| Method | Cases | Cases with findings | Wrong confirmations | Confirmed without measured fix |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, data in report["methods"].items():
        lines.append(
            f"| {safe(method)} | {data['cases']} | {data['cases_with_findings']} | {data['wrong_confirmation_count']} | {data['confirmed_without_measured_fix']} |"
        )
    lines += [
        "",
        "## Cases",
        "",
        "| Source record | Method | Case | Categories |",
        "|---:|---|---|---|",
    ]
    for item in report["findings"]:
        row = item["row"]
        lines.append(
            f"| {item['source_record']} | {safe(row['method'])} | {safe(row['case'])} | {', '.join(item['categories'])} |"
        )
    lines += [
        "",
        "Source record numbers refer to the normalized `rows.json` array (one-based).",
        "Full score evidence is in `report.json`. Source file and truth hashes are recorded there.",
        "This inventory does not establish why an error happened or evaluate new fault classes.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory already exists")
    raw = args.input.read_bytes()
    text = raw.decode()
    records = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    cases = load_cases(args.split)
    report = audit(records, cases)
    report.update(
        split=args.split,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        truth_sha256=hashlib.sha256(
            json.dumps({c.id: c.truth for c in cases}, sort_keys=True).encode()
        ).hexdigest(),
    )
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "rows.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.out / "README.md").write_text(markdown(report))
    print(f"Saved failure inventory to {args.out}")


if __name__ == "__main__":
    main()
