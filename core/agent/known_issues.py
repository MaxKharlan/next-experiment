"""B15: the known-issues step: Tavily searches built from the job config and symptom.

The code writes the queries from the model name, the recipe and the symptom, never
from dataset examples. The results go into the case as known_issues, and a
hypothesis may cite their URLs in its sources. Page text is data for the model:
instructions inside it change nothing, and the code drops any source URL that the
search did not return.
"""

from __future__ import annotations

from typing import Any

from core.connectors.tavily import Tavily

SNIPPET_CHARS = 400


def build_queries(job: dict[str, Any], symptom: str, max_words: int = 12) -> list[str]:
    model = job.get("model", "")
    hp = job.get("hyperparameters", {})
    short = " ".join(symptom.split()[:max_words])
    recipe = (
        f"LoRA fine-tuning learning_rate {hp.get('learning_rate', '')} "
        f"{hp.get('n_epochs', '')} epochs rank {hp.get('lora_r', '')} common problems"
    )
    return [f"{model} LoRA fine-tuning {short}", recipe]


def find_known_issues(
    tavily: Tavily,
    investigation_id: str,
    job: dict[str, Any],
    symptom: str,
    max_sources: int = 6,
) -> dict[str, Any]:
    """Search, merge the results by URL and keep the best-scored ones."""
    queries = build_queries(job, symptom)
    before = tavily.spent.get(investigation_id, 0)
    found: dict[str, dict[str, Any]] = {}
    for query in queries:
        reply = tavily.search(query, investigation_id=investigation_id)
        for r in reply.get("results", []):
            url = r.get("url")
            if url and (url not in found or r.get("score", 0) > found[url]["score"]):
                found[url] = {
                    "title": r.get("title", ""),
                    "url": url,
                    "snippet": (r.get("content") or "")[:SNIPPET_CHARS],
                    "score": r.get("score", 0.0),
                    "query": query,
                }
    sources = sorted(found.values(), key=lambda s: s["score"], reverse=True)
    return {
        "queries": queries,
        "sources": sources[:max_sources],
        "credits": tavily.spent.get(investigation_id, 0) - before,
    }
