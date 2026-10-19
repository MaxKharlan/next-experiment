"""B3: Tavily Search and Extract with a storage cache and credit limits.

Responses are cached in storage under <cache_prefix>/<sha256 of endpoint and
params>.json, so a rerun of an investigation costs no credits. Credits come from the
response's usage field, or from the documented price when usage is missing. Every
investigation has its own credit limit and the ledger has a persistent total
budget; a call that would pass either raises before it is sent. Unknown requests keep their
reservation and cannot be automatically retried. Local storage selects a SQLite
ledger; other storage requires an explicit ledger_path shared by all callers.
Only model names, library names and the symptom go to Tavily, never dataset examples.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from core.config import Settings, get_settings
from core.connectors.storage import LocalStorage, Storage
from core.db.credits import CreditBudgetError, CreditLedger, amount

API = "https://api.tavily.com"


TavilyBudgetError = CreditBudgetError


def search_credits(search_depth: str) -> float:
    if search_depth not in {"basic", "advanced"}:
        raise ValueError("Unsupported search depth")
    return 2 if search_depth == "advanced" else 1


def extract_credits(n_urls: int, extract_depth: str) -> float:
    # 1 credit per 5 successful URLs on basic, 2 on advanced
    return math.ceil(n_urls / 5) * search_credits(extract_depth)


@dataclass
class Tavily:
    api_key: str
    storage: Storage
    cache_prefix: str = "tavily-cache"
    max_credits_per_investigation: float = 10
    credit_budget: float = 200
    client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=30))
    spent: dict[str, float] = field(default_factory=dict)  # investigation -> credits

    ledger_path: Path | None = None
    ledger: CreditLedger = field(init=False)

    def __post_init__(self):
        if self.ledger_path is None:
            if not isinstance(self.storage, LocalStorage):
                raise ValueError(
                    "Nonlocal Tavily storage requires an explicit shared ledger_path"
                )
            self.ledger_path = self.storage.root / "tavily-credits.sqlite3"
        self.ledger = CreditLedger(
            self.ledger_path, self.credit_budget, self.max_credits_per_investigation
        )

    @classmethod
    def from_settings(cls, storage: Storage, s: Settings | None = None) -> Tavily:
        s = s or get_settings()
        if not s.tavily_api_key:
            raise ValueError("TAVILY_API_KEY must be set")
        return cls(
            api_key=s.tavily_api_key,
            storage=storage,
            cache_prefix=s.tavily_cache_prefix,
            max_credits_per_investigation=s.tavily_max_credits_per_investigation,
            credit_budget=s.tavily_credit_budget,
            ledger_path=Path(s.tavily_ledger_path) if s.tavily_ledger_path else None,
        )

    @property
    def total_credits(self) -> float:
        return self.ledger.snapshot()["committed_credits"]

    def search(
        self,
        query: str,
        *,
        investigation_id: str,
        max_results: int = 5,
        search_depth: str = "basic",
        **extra: Any,
    ) -> dict[str, Any]:
        if extra.get("auto_parameters"):
            raise ValueError(
                "Automatic search depth is incompatible with fixed credit reservations"
            )
        params = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            **extra,
        }
        return self._call(
            "search", params, investigation_id, search_credits(search_depth)
        )

    def extract(
        self, urls: list[str], *, investigation_id: str, extract_depth: str = "basic"
    ) -> dict[str, Any]:
        params = {"urls": urls, "extract_depth": extract_depth}
        price = extract_credits(len(urls), extract_depth)
        return self._call("extract", params, investigation_id, price)

    def _call(
        self, endpoint: str, params: dict[str, Any], investigation_id: str, price: float
    ) -> dict[str, Any]:
        digest = hashlib.sha256(
            json.dumps([endpoint, params], sort_keys=True).encode()
        ).hexdigest()
        key = f"{self.cache_prefix}/{digest}.json"
        if self.storage.exists(key):
            return {**json.loads(self.storage.get_text(key)), "cached": True}

        self.ledger.reserve(digest, investigation_id, price)
        response = self.client.post(
            f"{API}/{endpoint}",
            json={**params, "include_usage": True},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage")
        credits = usage.get("credits") if isinstance(usage, dict) else None
        credits = price if credits is None else amount(credits)
        self.ledger.settle(digest, credits)
        # Session delta for reports; admission uses the durable ledger, including holds.
        self.spent[investigation_id] = self.spent.get(investigation_id, 0) + credits
        self.storage.put_text(key, json.dumps(data))
        return {**data, "cached": False}
