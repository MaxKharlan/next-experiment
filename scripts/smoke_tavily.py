"""B3/B15 smoke test: known issues from Tavily for the fault-6 case, then hypotheses.

Two basic searches (2 Tavily credits; a rerun comes from the cache in
artifacts/tavily-cache/ and costs nothing) and one Super call (about $0.002).
Prints the queries, the sources and the hypotheses with the sources they cite.
"""

from __future__ import annotations

from core.agent.known_issues import find_known_issues
from core.agent.router import Router
from core.agent.steps import propose_hypotheses
from core.connectors.storage import LocalStorage
from core.connectors.tavily import Tavily
from core.connectors.token_factory import TokenFactory
from core.telemetry.calls import CallLog
from scripts.smoke_agent_steps import CASE


def main() -> None:
    tavily = Tavily.from_settings(LocalStorage("artifacts"))
    found = find_known_issues(tavily, "smoke-tavily", CASE["job"], CASE["symptom"])
    print(f"Queries ({found['credits']} credits):")
    for q in found["queries"]:
        print(f"  {q}")
    print("Sources:")
    for s in found["sources"]:
        print(f"  {s['score']:.2f} {s['title']} — {s['url']}")

    case = {
        **CASE,
        "known_issues": [
            {k: s[k] for k in ("title", "url", "snippet")} for s in found["sources"]
        ],
    }
    router = Router(TokenFactory.from_settings(), CallLog("smoke-tavily"))
    raw = {}
    structured = router.structured

    def keep_raw(*args, **kwargs):
        raw["reply"] = structured(*args, **kwargs)
        return raw["reply"]

    router.structured = keep_raw  # to show the sources before the code filters them
    print("Hypotheses (Super):")
    for h in propose_hypotheses(router, case):
        print(f"  {h.id} {h.fault_class} p={h.prior:.2f}: {h.rationale}")
        for url in h.sources:
            print(f"     source: {url}")
    cited = [u for h in raw["reply"].hypotheses for u in h.sources]
    print(f"URLs Super cited before filtering: {cited or 'none'}")
    print(f"Super cost ${router.log.cost_usd:.4f}")


if __name__ == "__main__":
    main()
