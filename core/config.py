from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    token_factory_api_key: str = ""
    token_factory_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    nemotron_nano_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    nemotron_super_model: str = "nvidia/nemotron-3-super-120b-a12b"
    nemotron_ultra_model: str = "nvidia/Nemotron-3-Ultra-550b-a55b"

    hf_token: str = ""

    storage_backend: str = "local"
    artifacts_dir: str = "./artifacts"
    database_url: str = "sqlite:///./artifacts/next_experiment.db"

    tavily_api_key: str = ""

    default_budget_usd: float = 5.0
    approval_threshold_usd: float = 1.0
    public_demo: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
