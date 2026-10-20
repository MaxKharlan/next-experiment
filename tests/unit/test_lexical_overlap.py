import json
import shutil
from pathlib import Path

import pytest

from bench.lexical_overlap_controls import verify
from core.eval.data import Example
from core.probes.level0.overlap import audit_files, find_candidates


def example(text, *, answer="label", system=None, line=1):
    prompt = [{"role": "user", "content": text}]
    if system:
        prompt.insert(0, {"role": "system", "content": system})
    return Example(line, prompt, answer)


def test_unicode_whitespace_exact_and_labels_do_not_control_detection():
    result = find_candidates(
        [example("Check ＡＢＣ", answer="a")],
        [example("CHECK  abc", answer="b")],
    )
    assert result["candidate_pairs"] == [
        {
            "train_line": 1,
            "validation_line": 1,
            "kind": "normalized_exact",
            "similarity": 1.0,
            "reference_answers_equal": False,
        }
    ]


def test_context_roles_and_short_prompts_do_not_match_fuzzily():
    text = "Please check the delivery status of my replacement payment card."
    assert not find_candidates(
        [example(text, system="Do not cancel")],
        [example(text, system="Cancel")],
    )["candidate_pairs"]
    assert not find_candidates([example("refund 100")], [example("refund 200")])[
        "candidate_pairs"
    ]
    prior_assistant = Example(
        1,
        [{"role": "assistant", "content": text}, {"role": "user", "content": "Yes"}],
        "label",
    )
    prior_user = Example(
        1,
        [{"role": "user", "content": text}, {"role": "user", "content": "Yes"}],
        "label",
    )
    assert not find_candidates([prior_assistant], [prior_user])["candidate_pairs"]


def test_surface_candidate_and_unique_validation_denominator():
    text = "Please check the delivery status of my replacement payment card."
    result = find_candidates(
        [example(text, line=2), example(text, line=7)],
        [example(text[:-1] + "!", line=4), example("Something else", line=8)],
    )
    assert len(result["candidate_pairs"]) == 2
    assert result["validation_lines"] == [4]
    assert result["validation_candidate_share"] == 0.5
    assert all(p["kind"] == "character_5gram" for p in result["candidate_pairs"])


def test_numeric_change_is_an_explicit_false_positive_limitation():
    text = "Please transfer exactly 100 dollars from my savings account to my checking account."
    assert find_candidates([example(text)], [example(text.replace("100", "200"))])[
        "candidate_pairs"
    ]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 1])
def test_search_limit_fails_explicitly(limit):
    with pytest.raises(ValueError, match="max_pairs"):
        find_candidates(
            [example("a")],
            [example("a", line=1), example("b", line=2)],
            max_pairs=limit,
        )


def test_invalid_prompt_or_duplicate_lines_rejected():
    with pytest.raises(ValueError, match="nonempty user"):
        find_candidates([example(" ")], [example("a")])
    with pytest.raises(ValueError, match="unique"):
        find_candidates([example("a"), example("b")], [example("a")])


def test_file_audit_keeps_source_line_numbers_and_hashes(tmp_path):
    train, valid = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    row = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "label"},
            ]
        }
    )
    train.write_text("\n" + row + "\n")
    valid.write_text(row + "\n")
    result = audit_files(train, valid)
    assert result["candidate_pairs"][0]["train_line"] == 2
    assert result["candidate_pairs"][0]["validation_line"] == 1
    assert result["dataset_sha256"]["train"] != result["dataset_sha256"]["validation"]
    with pytest.raises(ValueError, match="different files"):
        audit_files(train, train)


CONTROL_ROOT = Path("docs/results/lexical-overlap-controls-20261020-v2")


def test_frozen_controls_preserve_false_positives_and_paraphrase_misses():
    result = verify(CONTROL_ROOT)
    assert result["confusion"] == {"tp": 7, "fp": 4, "tn": 7, "fn": 4}
    paraphrases = [p for p in result["pairs"] if p["category"] == "paraphrase"]
    assert len(paraphrases) == 4
    assert all(p["confusion"] == "fn" for p in paraphrases)


@pytest.mark.parametrize("filename", ["controls.json", "report.json"])
def test_frozen_control_tampering_rejected(tmp_path, filename):
    root = tmp_path / "bundle"
    shutil.copytree(CONTROL_ROOT, root)
    path = root / filename
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        verify(root)
