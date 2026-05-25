from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    openai_api_key: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    chunk_size: int = 512
    chunk_overlap: int = 64
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
