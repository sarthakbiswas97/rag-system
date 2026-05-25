import pytest

from rag.config import Settings


@pytest.fixture()
def settings() -> Settings:
    return Settings(openai_api_key="test-key")
