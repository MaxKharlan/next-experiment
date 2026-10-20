"""Execute one frozen REST study after the ongoing MPS scorer releases its lock."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

from apps.api.code_investigations import verify_prepared
from core.eval.runtime import file_digest

ROOT = Path("docs/results/rest-code-investigation-20261020")
LEDGER = Path("artifacts/remaining-queue-20261020/ticket-ledger-v2")
BASE = "http://127.0.0.1:8767"


def save(name, value):
    path = ROOT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def ready():
    state = json.loads((LEDGER / "state.json").read_text())
    if len(state) != 12:
        return False
    for name in state:
        path = LEDGER / "scores" / f"{name}.json"
        if not path.exists() or not json.loads(path.read_text()).get("complete"):
            return False
    with (LEDGER.parent / "ticket-ledger-v2.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def start(env, log):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "apps.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8767",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError("API exited during startup; see local server log")
            try:
                if httpx.get(BASE + "/api/capabilities", timeout=1).status_code == 200:
                    return process
            except httpx.RequestError:
                pass
            time.sleep(0.5)
        raise TimeoutError("API did not become ready")
    except BaseException:
        process.terminate()
        process.wait(timeout=30)
        raise


def stop(process):
    process.terminate()
    process.wait(timeout=30)


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if file_digest(ROOT / "prepared.json") != protocol["prepared_sha256"]:
        raise ValueError("Frozen preparation changed")
    prepared = json.loads((ROOT / "prepared.json").read_text())
    request = json.loads((ROOT / "request.json").read_text())
    if request != prepared["request"]:
        raise ValueError("Frozen request changed")
    if (ROOT / "submission-intent.json").exists():
        raise ValueError("An execution attempt already exists; no automatic retry")
    deadline = time.monotonic() + 6 * 3600
    save("execution-status.json", {"status": "waiting_for_twelve_mps_scores"})
    while not ready():
        if time.monotonic() > deadline:
            raise TimeoutError("MPS dependency did not complete within six hours")
        time.sleep(30)
    if not (LEDGER / "decision.json").exists():
        with (ROOT / "ledger-analysis.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.execute_ticket_ledger_v2",
                    "analyze",
                    "--root",
                    str(LEDGER),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    verify_prepared(prepared)
    env = os.environ.copy()
    values = dotenv_values("/Users/harlanmax/nebiusXnvidia/.env")
    if not values.get("TOKEN_FACTORY_API_KEY"):
        raise ValueError("Missing provider credential")
    env["TOKEN_FACTORY_API_KEY"] = values["TOKEN_FACTORY_API_KEY"]
    env.update(
        STORAGE_BACKEND="local",
        INVESTIGATION_BACKEND="sqlite",
        ARTIFACTS_DIR="artifacts/rest-code-trace-20261020",
        DATABASE_URL="sqlite:///artifacts/rest-code-trace-20261020/state.sqlite3",
        PUBLIC_DEMO="false",
        NEXT_EXPERIMENT_SPENDING_LEDGER="artifacts/remaining-queue-20261020/spending.sqlite3",
        NEXT_EXPERIMENT_SPENDING_CAP_USD="25",
        PYTHONUNBUFFERED="1",
    )
    Path(env["ARTIFACTS_DIR"]).mkdir(exist_ok=False)
    headers = {"Idempotency-Key": protocol["idempotency_key"]}
    with (ROOT / "server.log").open("w") as log:
        server = start(env, log)
        try:
            save(
                "submission-intent.json",
                {
                    "endpoint": protocol["endpoint"],
                    "key": protocol["idempotency_key"],
                    "attempts": 1,
                },
            )
            response = httpx.post(
                protocol["endpoint"], json=request, headers=headers, timeout=120
            )
            save(
                "accepted.json",
                {"http_status": response.status_code, "body": response.json()},
            )
            response.raise_for_status()
            inv_id = response.json()["id"]
            save("execution-status.json", {"status": "running", "id": inv_id})
            with (
                httpx.stream(
                    "GET", BASE + f"/api/investigations/{inv_id}/events", timeout=120
                ) as stream,
                (ROOT / "events.sse").open("w") as events,
            ):
                stream.raise_for_status()
                for line in stream.iter_lines():
                    events.write(line + "\n")
                    events.flush()
            report = httpx.get(
                BASE + f"/api/investigations/{inv_id}/report", timeout=30
            )
            report.raise_for_status()
            save("report.json", report.json())
        finally:
            stop(server)
        server = start(env, log)
        try:
            after = httpx.get(BASE + f"/api/investigations/{inv_id}/report", timeout=30)
            after.raise_for_status()
            replay = httpx.post(
                protocol["endpoint"], json=request, headers=headers, timeout=30
            )
            replay.raise_for_status()
            equal = after.json() == report.json()
            reused = replay.json() == {"id": inv_id, "reused": True}
            save(
                "restart-verification.json",
                {
                    "report_equal": equal,
                    "idempotent_post": replay.json(),
                    "reused_without_launch": reused,
                },
            )
            if not equal or not reused:
                raise ValueError("Restart/idempotency verification failed")
            save(
                "execution-status.json",
                {
                    "status": "trace_complete",
                    "id": inv_id,
                    "outcome": report.json().get("diagnosis_status"),
                    "restart_verified": True,
                },
            )
        finally:
            stop(server)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save(
            "execution-error.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "automatic_retry": False,
            },
        )
        raise
