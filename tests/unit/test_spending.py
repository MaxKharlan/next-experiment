from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from core.config import Settings
from core.connectors.token_factory import TokenFactory
from core.db.spending import BudgetExceeded, SpendingLedger


def test_concurrent_reservations_and_restart(tmp_path):
    path = tmp_path / "budget.sqlite3"
    SpendingLedger(path, 1)

    def attempt(i):
        ledger = SpendingLedger(path, 1)
        try:
            ledger.reserve(str(i), "job", 0.6)
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(attempt, [0, 1])) == 1
    ledger = SpendingLedger(path, 1)
    assert ledger.snapshot()["committed_usd"] == 0.6
    op = ledger.snapshot()["operations"][0]["id"]
    with pytest.raises(ValueError, match="already recorded"):
        ledger.reserve(op, "job", 0.6)
    ledger.settle(op, 0.2)
    assert ledger.snapshot()["available_usd"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="differs"):
        SpendingLedger(path, 10)


def test_overrun_is_recorded_and_blocks_new_work(tmp_path):
    ledger = SpendingLedger(tmp_path / "budget.db", 1)
    ledger.reserve("a", "job", 0.5)
    ledger.settle("a", 1.2)
    assert ledger.snapshot()["committed_usd"] == 1.2
    with pytest.raises(BudgetExceeded):
        ledger.reserve("b", "job", 0.01)


class Client:
    def __init__(self, ledger, fail=False):
        self.ledger, self.fail, self.calls = ledger, fail, 0
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **options):
        assert options == {"max_retries": 0}
        return self

    def create(self, **kwargs):
        assert self.ledger.snapshot()["operations"][-1]["status"] == "pending"
        self.calls += 1
        if self.fail:
            raise TimeoutError("unknown provider outcome")
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )


def test_inference_reserves_before_send_and_holds_on_timeout(tmp_path):
    ledger = SpendingLedger(tmp_path / "budget.db", 1)
    client = Client(ledger, fail=True)
    tf = TokenFactory(client, Settings(), spending=ledger)
    with pytest.raises(TimeoutError):
        tf.chat(tf.model("super"), [{"role": "user", "content": "hello"}])
    assert ledger.snapshot()["operations"][0]["status"] == "pending"
    client.fail = False
    call = tf.chat(tf.model("super"), [{"role": "user", "content": "hello"}])
    assert ledger.snapshot()["operations"][1]["actual_usd"] == call.cost_usd
    tiny = SpendingLedger(tmp_path / "tiny.db", 1e-8)
    tf.spending = tiny
    with pytest.raises(BudgetExceeded):
        tf.chat(tf.model("super"), [])
    assert client.calls == 2
