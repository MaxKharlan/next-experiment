"""Private, fail-closed balance check against the Token Factory billing console."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from decimal import Decimal, InvalidOperation
from http.cookies import SimpleCookie
from pathlib import Path

import httpx

ORIGIN = "https://tokenfactory.nebius.com"
BALANCE_PATH = "/api-mfe/billing/gateway/root/customers/getBalance"
CACHE_SECONDS = 30


class ProviderBalance:
    def __init__(self, config_path=None, block_path=None):
        self.config_path = Path(
            config_path
            or os.environ.get("JUDGE_BILLING_CONFIG", "/data/billing-session.json")
        )
        self.block_path = Path(block_path or "/data/billing-block.json")
        self.lock = threading.RLock()
        self.checked_at = 0.0
        self.value = None

    def _fetch(self):
        config = json.loads(self.config_path.read_text())
        contract = config["contract_id"]
        if not isinstance(contract, str) or not contract:
            raise ValueError("Missing billing contract")
        headers = config["headers"]
        # All requests are to one fixed HTTPS origin. Never follow an auth redirect
        # or forward a captured credential to a caller-supplied destination.
        allowed = {
            "cookie",
            "x-csrf-token",
            "x-requested-with",
            "user-agent",
            "x-nebius-tenant-id",
        }
        headers = {k.lower(): v for k, v in headers.items() if k.lower() in allowed}
        parsed_cookies = SimpleCookie()
        parsed_cookies.load(headers.pop("cookie", ""))
        cookies = httpx.Cookies()
        for name, entry in parsed_cookies.items():
            cookies.set(name, entry.value, domain="tokenfactory.nebius.com", path="/")
        headers.update({"origin": ORIGIN, "referer": ORIGIN + "/"})
        with httpx.Client(
            base_url=ORIGIN,
            timeout=12,
            follow_redirects=False,
            headers=headers,
            cookies=cookies,
        ) as client:
            response = client.post(BALANCE_PATH, json={"contractId": contract})
            error_data = (
                response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else {}
            )
            if response.status_code in (401, 403) or (
                isinstance(error_data, dict)
                and error_data.get("code") == "EBADCSRFTOKEN"
            ):
                # One read-only CSRF refresh; this never signs in or makes a payment.
                page = client.get("/")
                page.raise_for_status()
                match = re.search(r'"csrfToken"\s*:\s*("(?:\\.|[^"\\])*")', page.text)
                if not match:
                    raise ValueError("Billing session needs renewal")
                client.headers["x-csrf-token"] = json.loads(match.group(1))
                response = client.post(BALANCE_PATH, json={"contractId": contract})
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise TypeError("Invalid billing response")
            if data.get("statusCode", 200) >= 400:
                raise ValueError("Billing request failed")
            if data.get("currency", "USD") != "USD":
                raise ValueError("Billing currency must be USD")
            raw = data["balance"]
            if isinstance(raw, bool):
                raise TypeError("Invalid balance")
            value = Decimal(str(raw))
            if not value.is_finite():
                raise ValueError("Invalid balance")
            return max(0.0, float(value))

    def read(self, *, force=False):
        with self.lock:
            if not force and time.monotonic() - self.checked_at < CACHE_SECONDS:
                return self.value
            try:
                value = self._fetch()
                if not math.isfinite(value):
                    raise ValueError("Invalid balance")
                if self.block_path.exists():
                    blocked = json.loads(self.block_path.read_text())
                    # A 402 overrides a lagging positive balance until a top-up
                    # exceeds the last observed amount, or the block is reconciled.
                    if value <= blocked["balance_at_error"]:
                        self.block_path.write_text(
                            json.dumps({"balance_at_error": value})
                        )
                        value = 0.0
                    else:
                        self.block_path.unlink()
                self.value = value
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                InvalidOperation,
                httpx.HTTPError,
            ):
                self.value = None
            self.checked_at = time.monotonic()
            return self.value

    def exhausted(self):
        with self.lock:
            previous = self.value or 0.0
            if self.block_path.exists():
                previous = max(
                    previous,
                    json.loads(self.block_path.read_text())["balance_at_error"],
                )
            self.block_path.parent.mkdir(parents=True, exist_ok=True)
            self.block_path.write_text(json.dumps({"balance_at_error": previous}))
            self.block_path.chmod(0o600)
            self.value = 0.0
            self.checked_at = time.monotonic()


def public_availability(wallet, balance, run_cap):
    """Expose the effective demo allowance, never account or session details."""
    used = sum(r["reserved_usd"] for r in wallet["runs"].values())
    holds = sum(
        r["reserved_usd"] for r in wallet["runs"].values() if r["status"] != "complete"
    )
    active = any(r["status"] == "running" for r in wallet["runs"].values())
    remaining = (
        None
        if balance is None
        else max(0, min(wallet["cap_usd"] - used, balance - holds))
    )
    reason = (
        "unavailable"
        if remaining is None
        else "busy"
        if active
        else "exhausted"
        if remaining < run_cap
        else "available"
    )
    return {
        "available": reason == "available",
        "busy": active,
        "status": reason,
        "allocation_usd": wallet["cap_usd"],
        "run_allowance_usd": run_cap,
        # Floor instead of rounding up to money that cannot be spent.
        "funded_remaining_usd": None
        if remaining is None
        else math.floor(remaining * 100) / 100,
    }
