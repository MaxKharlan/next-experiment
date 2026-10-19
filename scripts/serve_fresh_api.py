"""Run one loopback API worker using credentials from an explicit env file.

Only Token Factory credentials are imported. SQLite state, disabled weight probes
and disabled web search keep the fresh run within its recorded resource budget.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.config import Settings, get_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    source = Settings(_env_file=args.env_file)
    if not source.token_factory_api_key:
        raise ValueError("Token Factory credentials unavailable")
    os.environ.update(
        TOKEN_FACTORY_API_KEY=source.token_factory_api_key,
        TOKEN_FACTORY_BASE_URL=source.token_factory_base_url,
        INVESTIGATION_BACKEND="sqlite",
        DATABASE_URL=f"sqlite:///{args.database.resolve()}",
        STORAGE_BACKEND="local",
        ARTIFACTS_DIR="./artifacts",
        PUBLIC_DEMO="false",
        TAVILY_API_KEY="",
        NEXT_EXPERIMENT_WEIGHTS="0",
        HYPOTHESIS_SAMPLES="1",
        HF_HUB_OFFLINE="1",
    )
    # Model routing is the repository's frozen default, not an unrelated .env.
    for field in (
        "nemotron_nano_model",
        "nemotron_super_model",
        "nemotron_ultra_model",
    ):
        os.environ[field.upper()] = Settings.model_fields[field].default
    get_settings.cache_clear()
    del source
    import uvicorn

    uvicorn.run("apps.api.main:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
