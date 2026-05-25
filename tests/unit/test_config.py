from rag.config import Settings, get_settings


class TestSettings:
    def test_defaults(self) -> None:
        settings = Settings(openai_api_key="test")
        assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
        assert settings.embedding_batch_size == 64
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 64
        assert settings.llm_model == "gpt-4o-mini"
        assert settings.llm_temperature == 0.1

    def test_override_via_constructor(self) -> None:
        settings = Settings(openai_api_key="test", chunk_size=256)
        assert settings.chunk_size == 256

    def test_get_settings_returns_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)
