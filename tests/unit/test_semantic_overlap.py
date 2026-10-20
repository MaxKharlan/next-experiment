import pytest

from core.probes.level0.semantic_overlap import compare_prompts


def prompt(system, user):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def test_identical_vectors_do_not_override_system_context_or_roles():
    left = prompt("Translate to French", "hello")
    right = prompt("Translate to German", "hello")
    assert compare_prompts(left, right, [1, 0], [1, 0])["kind"] == "context_guard"
    right = [{"role": "user", "content": "hello"}]
    assert not compare_prompts(left, right, [1, 0], [1, 0])["candidate"]


def test_normalized_exact_and_cosine_paths_validate_different_evidence():
    left = prompt("Classify", "  Hello   WORLD ")
    right = prompt("Classify", "hello world")
    assert compare_prompts(left, right, None, None)["kind"] == "normalized_exact"
    right = prompt("Classify", "something different")
    assert not compare_prompts(left, right, [1, 0], [0, 1])["candidate"]
    with pytest.raises(ValueError):
        compare_prompts(left, right, [0, 0], [1, 0])
    with pytest.raises(ValueError):
        compare_prompts(left, right, [1, 0], [1, 0], threshold=float("nan"))
