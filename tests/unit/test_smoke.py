from core.config import Settings


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.token_factory_base_url.startswith("https://")
    assert s.storage_backend == "local"
