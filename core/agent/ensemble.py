"""Aggregate independent hypothesis calls by fault class."""

from __future__ import annotations

from core.agent.steps import propose_hypotheses


def combine(draws):
    if not draws:
        raise ValueError("At least one hypothesis draw is required")
    groups = {}
    for draw_index, draw in enumerate(draws):
        for hypothesis in draw:
            group = groups.setdefault(hypothesis.fault_class, [])
            group.append((draw_index, hypothesis))
    merged = []
    support = {}
    for fault, entries in groups.items():
        representative = max(entries, key=lambda item: item[1].prior)[1]
        # Missing classes contribute zero. Duplicate entries within one draw
        # share its probability mass, but count only once toward support.
        prior = sum(h.prior for _, h in entries) / len(draws)
        sources = sorted({source for _, h in entries for source in h.sources})
        merged.append(
            representative.model_copy(update={"prior": prior, "sources": sources})
        )
        support[fault] = len({index for index, _ in entries})
    merged.sort(key=lambda h: (-h.prior, h.fault_class))
    return [
        h.model_copy(update={"id": f"h{i}"}) for i, h in enumerate(merged, 1)
    ], support


def sample_hypotheses(router, case, *, mode, samples, budget_usd):
    if type(samples) is not int or not 2 <= samples <= 5:
        raise ValueError("Ensemble sampling requires 2 to 5 draws")
    draws = []
    for _ in range(samples):
        if router.log.cost_usd >= budget_usd:
            raise ValueError("Hypothesis sampling exhausted the investigation budget")
        # A fresh call receives the same case, without previous model answers.
        draws.append(propose_hypotheses(router, case, mode=mode))
    hypotheses, support = combine(draws)
    return hypotheses, {
        "kind": "hypothesis_sampling",
        "samples": samples,
        "aggregation": "mean class probability; missing classes contribute zero",
        "support_counts": support,
        "draws": [[h.model_dump() for h in draw] for draw in draws],
    }
