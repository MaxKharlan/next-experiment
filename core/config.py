from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    token_factory_api_key: str = ""
    token_factory_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    nemotron_nano_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    nemotron_super_model: str = "nvidia/nemotron-3-super-120b-a12b"
    nemotron_ultra_model: str = "nvidia/Nemotron-3-Ultra-550b-a55b"

    hf_token: str = ""

    storage_backend: Literal["local", "s3"] = "local"
    artifacts_dir: str = "./artifacts"
    investigation_backend: Literal["storage", "sqlite"] = "storage"
    database_url: str = "sqlite:///./artifacts/next_experiment.db"

    s3_endpoint_url: str = "https://storage.eu-north1.nebius.cloud"
    s3_region: str = "eu-north1"
    s3_bucket: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""

    tavily_api_key: str = ""
    tavily_max_credits_per_investigation: float = 10
    tavily_credit_budget: float = 200
    tavily_cache_prefix: str = "tavily-cache"
    tavily_ledger_path: str | None = None

    default_budget_usd: float = 5.0
    approval_threshold_usd: float = 1.0
    public_demo: bool = False
    hypothesis_samples: int = Field(default=1, ge=1, le=5)


@lru_cache
def get_settings() -> Settings:
    return Settings()
