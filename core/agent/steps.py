"""B13: the model steps of an investigation: hypotheses, probe cards, the verdict.

Each step renders a versioned prompt from core/agent/prompts, calls the model the
router picks for the step and returns a validated object. Decisions stay in code:
the verdict status comes from the causal check, and hypotheses outside the catalog
for the mode are dropped.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Any
from urllib.parse import urlsplit

from core.agent.catalog import allowed_classes, catalog_text
from core.agent.router import Router
from core.agent.schemas import (
    Hypothesis,
    HypothesisList,
    ProbeInterpretation,
    Verdict,
)
from core.interventions.pairs import Intervention
from core.interventions.stats import CausalResult

PROMPTS = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.name}-{self.version}"

    def render(self, **values: str) -> str:
        return Template(self.text).substitute(values)


def load_prompt(name: str, version: str = "v1") -> Prompt:
    text = (PROMPTS / f"{name}.{version}.md").read_text(encoding="utf-8")
    return Prompt(name, version, text)


def _user(title: str, payload: Any) -> dict[str, str]:
    return {"role": "user", "content": f"{title}:\n{json.dumps(payload, indent=2)}"}


def url_key(url: str) -> str:
    """Host and path without scheme, query, fragment or trailing slash.

    Anything before http(s):// is dropped: Super has cited "/https://..."."""
    found = re.search(r"https?://\S+", url)
    parts = urlsplit(found.group(0) if found else url.strip())
    return f"{parts.netloc.lower().removeprefix('www.')}{parts.path.rstrip('/')}"


def clean_hypotheses(
    found: list[Hypothesis], mode: str, known_urls: set[str] | None = None
) -> list[Hypothesis]:
    """Drop classes the mode cannot have and sources the search did not return,
    number the rest h1.. and cap priors at 1. A cited URL counts when it matches a
    known one up to query, fragment and trailing slash; the known URL is kept."""
    kept = [h for h in found if h.fault_class in allowed_classes(mode) | {"other"}]
    total = sum(h.prior for h in kept)
    scale = 1 / total if total > 1 else 1.0
    known = {url_key(u): u for u in known_urls or set()}
    return [
        h.model_copy(
            update={
                "id": f"h{i}",
                "prior": h.prior * scale,
                "sources": list(
                    dict.fromkeys(
                        known[url_key(u)] for u in h.sources if url_key(u) in known
                    )
                ),
            }
        )
        for i, h in enumerate(kept, start=1)
    ]


def propose_hypotheses(
    router: Router, case: dict[str, Any], mode: str = "A", version: str = "v2"
) -> list[Hypothesis]:
    """Super lists hypotheses for a case: symptom, job config, metrics, findings
    and known issues found by Tavily."""
    prompt = load_prompt("hypotheses", version)
    messages = [
        {"role": "system", "content": prompt.render(catalog=catalog_text(mode))},
        _user("Case", case),
    ]
    found = router.structured(
        "hypotheses", messages, HypothesisList, prompt_version=prompt.id
    )
    known = {s["url"] for s in case.get("known_issues", [])}
    return clean_hypotheses(found.hypotheses, mode, known)


def interpret_probe(
    router: Router,
    hypotheses: list[Hypothesis],
    probe_name: str,
    outcome: str,
    summary: dict[str, Any],
    version: str = "v1",
) -> ProbeInterpretation:
    """Super explains a probe outcome for its card; the code does the Bayes update."""
    prompt = load_prompt("interpret_probe", version)
    payload = {
        "hypotheses": [h.model_dump() for h in hypotheses],
        "probe_name": probe_name,
        "outcome": outcome,
        "summary": summary,
    }
    messages = [{"role": "system", "content": prompt.text}, _user("Check", payload)]
    card = router.structured(
        "interpret_probe", messages, ProbeInterpretation, prompt_version=prompt.id
    )
    return card.model_copy(update={"probe_name": probe_name, "outcome": outcome})


def write_verdict(
    router: Router,
    hypothesis: Hypothesis,
    target: Intervention,
    control: Intervention,
    result: CausalResult,
    base_hp: dict[str, Any],
    evidence: list[str],
    metric: str = "exact match on held-out examples",
    version: str = "v1",
) -> Verdict:
    """Ultra writes the summary; status, evidence and patch come from the code."""
    prompt = load_prompt("verdict", version)
    patch = target.diff(base_hp) if result.status == "confirmed" else ""
    payload = {
        "hypothesis": hypothesis.model_dump(),
        "metric": metric,
        "target": {"change": target.description, "metric_per_seed": target.metrics},
        "control": {"change": control.description, "metric_per_seed": control.metrics},
        "causal_check": asdict(result),
        "status": result.status,
        "evidence": evidence,
        "patch": patch,
    }
    messages = [{"role": "system", "content": prompt.text}, _user("Verdict", payload)]
    verdict = router.structured("verdict", messages, Verdict, prompt_version=prompt.id)
    return verdict.model_copy(
        update={
            "hypothesis_id": hypothesis.id,
            "status": result.status,
            "evidence": evidence,
            "patch": patch,
        }
    )
