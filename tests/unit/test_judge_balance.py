import json
import shlex

import httpx
import pytest

from apps.judge.balance import (
    BALANCE_PATH,
    ORIGIN,
    ProviderBalance,
    public_availability,
)
from apps.judge.service import BalanceCheckedLedger
from core.db.spending import BudgetExceeded
from scripts.configure_judge_billing import parse_capture
from tests.unit.test_judge import MemoryHub, StubBalance


def test_external_spending_caps_demo_and_zero_disables_admission():
    hub = MemoryHub(cap=20)
    hub.balance.value = 3
    assert hub.available()["funded_remaining_usd"] == 3
    hub.balance.value = 0
    assert hub.available()["status"] == "exhausted"
    assert hub.available()["funded_remaining_usd"] == 0
    with pytest.raises(BudgetExceeded):
        hub.claim("a", "key", "payload")
    assert hub.files["wallet.json"]["runs"] == {}
    hub.balance.value = None
    assert hub.available()["status"] == "unavailable"
    assert hub.available()["funded_remaining_usd"] is None
    with pytest.raises(BudgetExceeded):
        hub.claim("a", "key", "payload")
    assert hub.balance.forced == 2


def test_reservations_reduce_effective_allowance_without_exposing_account():
    wallet = {
        "cap_usd": 20,
        "runs": {
            "old": {"status": "complete", "reserved_usd": 2},
            "unknown": {"status": "failed", "reserved_usd": 1},
        },
    }
    assert public_availability(wallet, 100, 1)["funded_remaining_usd"] == 17
    result = public_availability(wallet, 5, 1)
    assert result["funded_remaining_usd"] == 4
    assert set(result) == {
        "available",
        "busy",
        "status",
        "allocation_usd",
        "run_allowance_usd",
        "funded_remaining_usd",
    }


def test_before_paid_guard_never_records_reservation_with_unknown_balance(tmp_path):
    balance = StubBalance(None)
    ledger = BalanceCheckedLedger(tmp_path / "spending.db", 1, balance)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("call", "test", 0.1)
    assert ledger.snapshot()["operations"] == []
    balance.value = 1
    ledger.reserve("call", "test", 0.1)
    balance.value = 0
    with pytest.raises(BudgetExceeded):
        ledger.ensure_available()
    ledger.record_provider_receipt(
        "call", request_id=None, response_id=None, outcome="http_error", http_status=402
    )
    assert balance.value == 0
    assert ledger.snapshot()["operations"][0]["status"] == "pending"


def make_provider(tmp_path, monkeypatch, handler):
    config = tmp_path / "session.json"
    config.write_text(
        json.dumps(
            {
                "contract_id": "contract-test",
                "headers": {"cookie": "session=test-only", "x-csrf-token": "test-only"},
            }
        )
    )
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    return ProviderBalance(config, tmp_path / "blocked.json")


def test_balance_cache_force_and_failed_refresh_discard_positive_value(
    tmp_path, monkeypatch
):
    replies = [{"balance": "12.34"}, {"balance": "0"}, {"balance": "NaN"}]
    calls = []

    def handler(request):
        calls.append(request.url)
        assert str(request.url) == ORIGIN + BALANCE_PATH
        assert json.loads(request.content) == {"contractId": "contract-test"}
        return httpx.Response(200, json=replies.pop(0))

    balance = make_provider(tmp_path, monkeypatch, handler)
    assert balance.read() == 12.34
    assert balance.read() == 12.34
    assert len(calls) == 1
    assert balance.read(force=True) == 0
    assert balance.read(force=True) is None
    assert balance.read() is None


def test_session_refresh_does_not_follow_auth_redirect(tmp_path, monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(403, json={})
        return httpx.Response(302, headers={"location": "https://other.example/login"})

    balance = make_provider(tmp_path, monkeypatch, handler)
    assert balance.read() is None
    assert len(requests) == 2
    assert all(r.url.host == "tokenfactory.nebius.com" for r in requests)


def test_402_block_survives_restart_until_observed_refill(tmp_path, monkeypatch):
    value = [5]
    balance = make_provider(
        tmp_path,
        monkeypatch,
        lambda request: httpx.Response(200, json={"balance": value[0]}),
    )
    assert balance.read() == 5
    balance.exhausted()
    restarted = ProviderBalance(balance.config_path, balance.block_path)
    assert restarted.read() == 0
    value[0] = 0
    assert restarted.read(force=True) == 0
    value[0] = 3
    assert restarted.read(force=True) == 3
    assert not restarted.block_path.exists()


def test_capture_is_parsed_as_data_and_rejects_other_destinations():
    command = shlex.join(
        [
            "curl",
            ORIGIN + BALANCE_PATH,
            "-H",
            "cookie: session=test-only",
            "-H",
            "x-csrf-token: $(never-execute)",
            "--data-raw",
            '{"contractId":"contract-test"}',
        ]
    )
    parsed = parse_capture(command)
    assert parsed["headers"]["x-csrf-token"] == "$(never-execute)"
    assert parsed["contract_id"] == "contract-test"
    with pytest.raises(ValueError):
        parse_capture(command.replace(ORIGIN, "https://other.example"))
    with pytest.raises(ValueError):
        parse_capture(command + "; touch /tmp/not-authorized")
