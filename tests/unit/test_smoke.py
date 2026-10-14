def test_settings_defaults():
    from core.config import get_settings

    s = get_settings()
    assert s.token_factory_base_url.startswith("https://")
    assert s.storage_backend == "local"
