# Browser training-label audit

GitHub Pages exposes “Inspect local training labels” independently of saved replays.
Paste a JSON request with `training_labels.reference`, `training_labels.observed`,
and optional `training_labels.eos_token_id`. Each row has `input_ids` and `labels`.
An optional request `title` accepts 1–120 characters. Unknown fields are rejected.
Click “Run audit in this browser” to compute the result locally. No request containing
these inputs is sent, and nothing is persisted. The live API retains its save workflow.

The reference must be unpadded, unshifted, answer-only labels: `-100` marks prompt
positions. The caller owns reference correctness. The audit compares supplied tensors;
it cannot assess semantic correctness or establish a cause of model behavior. Removed
EOS targets do not prove that padding logic caused the removal.

The browser port preserves the Python row/token limits, strict integer validation,
position-zero exclusion, and prompt/EOS applicability rules. Large integer IDs use
BigInt while JSON decimal/exponent literals are rejected as non-integers, matching
Pydantic StrictInt. Empty individual rows are accepted, as in Python.

## Verify locally

From the repository root, with Python 3.11/3.12 and Node installed:

```sh
uv sync
uv run pytest tests/unit/test_browser_label_audit.py tests/unit/test_agent_label_checks.py tests/unit/test_build_pages.py -q
uv run python -m scripts.build_pages --out _site
python3 -m http.server 8765 --directory _site
```

Open http://localhost:8765 and run the prefilled EOS example. Change observed labels
to match reference to see `aligned` / `preserved`. An invalid request must hide the
previous result and show the validation error. The parity test compares 1271 JSON
requests against the Python implementation, including randomized patterns and limits.

## Publish after review

Changes are prepared on `codex/browser-label-audit`. No merge or deployment is needed
for local verification. To submit this branch:

```sh
git push -u origin codex/browser-label-audit
gh pr create --base main --head codex/browser-label-audit --title "Run training label audits locally on GitHub Pages" --body "Adds a browser port of the Python label checks, local input validation and detailed results, with Python/JavaScript parity tests."
```

After reviewing and explicitly deciding to publish, merge that PR (replace NUMBER):

```sh
gh pr merge NUMBER --squash
gh run list --workflow pages.yml --branch main --limit 1
gh run watch RUN_ID --exit-status
```

The existing Pages workflow builds and deploys on changes to these files on main.
If a manual rerun is needed after merge:

```sh
gh workflow run pages.yml --ref main
```

The published site is https://maxkharlan.github.io/next-experiment/.
