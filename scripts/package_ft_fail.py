"""Package already-published FT-Fail evidence without running any benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    def read(path):
        path = Path(path)
        sources[str(path)] = digest(path)
        return json.loads(path.read_text())

    def write_rows(relative, rows):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )

    for split in ("dev", "heldout"):
        cases = []
        for path in sorted(Path("bench/cases", split).glob("*.json")):
            case = read(path)
            cases.append(
                {
                    "id": case["id"],
                    "split": split,
                    "task": case["task"],
                    "model": case["model"],
                    "injected_faults": case["truth"],
                    "case_json": json.dumps(case["case"], sort_keys=True),
                    "hyperparameters_json": json.dumps(
                        case["hyperparameters"], sort_keys=True
                    ),
                    "job_id": case["job_id"],
                    "saved_quality": case["heldout_metric"],
                    "source_manifest": str(path),
                    "source_sha256": sources[str(path)],
                }
            )
        assert len(cases) == (5 if split == "dev" else 8)
        write_rows(f"cases/{split}.jsonl", cases)
        rows = read(f"docs/results/failure-audit-20261018/{split}/rows.json")
        assert len(rows) == len(cases) * 4
        assert {r["case"] for r in rows} == {c["id"] for c in cases}
        write_rows(f"methods/{split}.jsonl", rows)
    base = Path("docs/results/semantic-audit-20261020")
    truth = {row["id"]: row for row in read(base / "constructed-truth.json")}
    audit = [{**row, **truth[row["id"]]} for row in read(base / "audit.json")]
    reference = read(base / "reference.json")
    assert len(audit) == 200 and len(reference) == 400
    assert {r["text"] for r in audit}.isdisjoint({r["text"] for r in reference})
    write_rows("semantic_audit/dev.jsonl", audit)
    write_rows("semantic_reference/dev.jsonl", reference)

    evidence = [
        Path("LICENSE"),
        Path("scripts/package_ft_fail.py"),
        Path("docs/benchmark.md"),
        Path("docs/semantic-audit-20261020.md"),
        Path("docs/semantic-data-controls-20261020.md"),
        Path("docs/logit-audit-20261020.md"),
        base / "protocol.json",
        base / "acceptance-amendment.json",
        base / "combined-results/report.json",
        base / "embedding-results/report.json",
        Path("docs/results/semantic-overlap-20261020/report.json"),
        Path("docs/results/semantic-overlap-20261020/controls.json"),
        Path("docs/results/distribution-controls-20261020/result.json"),
    ]
    evidence.extend(Path("bench/cases").glob("*/*.json"))
    for path in evidence:
        destination = output / "evidence" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        sources[str(path)] = digest(path)
        if path.suffix == ".md":

            def absolute_link(match, source_path=path):
                label, target = match.groups()
                if ":" in target or target.startswith("#"):
                    return match.group(0)
                linked = (source_path.parent / target).resolve()
                if not linked.exists() or not linked.is_relative_to(Path.cwd()):
                    return match.group(0)
                relative = linked.relative_to(Path.cwd()).as_posix()
                kind = "tree" if linked.is_dir() else "blob"
                return f"[{label}](https://github.com/MaxKharlan/next-experiment/{kind}/{revision}/{quote(relative)})"

            destination.write_text(
                re.sub(r"\[([^\]]+)\]\(([^)]+)\)", absolute_link, path.read_text())
            )
    for name in ("README.md", "LICENSES.md"):
        path = Path("docs/ft-fail-hub") / name
        shutil.copyfile(path, output / name)
        sources[str(path)] = digest(path)
    manifest = {
        "release": "2026-10-20",
        "repository": "https://github.com/MaxKharlan/next-experiment",
        "source_commit": revision,
        "packaging_change": "Existing relative links in evidence Markdown resolve to the pinned GitHub source revision.",
        "scope": "Completed historical benchmark and automatic development audit; no new inference or uncompleted final data.",
        "sources_sha256": sources,
        "files_sha256": {
            str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.out)
    print(json.dumps({"files": len(result["files_sha256"]), "out": str(args.out)}))
