"""Keep the legacy storage-backed API in one process on this host."""

import fcntl
import os
import threading
from pathlib import Path

_HANDLES = {}
_LOCK = threading.Lock()


def hold_single_worker(directory):
    path = Path(directory).resolve() / ".api-single-worker.lock"
    key = (os.getpid(), str(path))
    with _LOCK:
        if key in _HANDLES:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise ValueError(
                "Storage investigation state supports one API process; "
                "use INVESTIGATION_BACKEND=sqlite with a shared DATABASE_URL for multiple workers"
            ) from None
        _HANDLES[key] = handle  # The OS releases the lock on process exit.
