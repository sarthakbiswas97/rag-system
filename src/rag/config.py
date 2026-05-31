from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    openai_api_key: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_backend: str = "pytorch"  # "pytorch" or "onnx"
    embedding_onnx_provider: str = "CPUExecutionProvider"
    embedding_batch_size: int = 64
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"
    qdrant_shard_number: int = 6
    qdrant_replication_factor: int = 1
    chunk_size: int = 512
    chunk_overlap: int = 64
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranking: bool = True
    enable_sparse_search: bool = True
    sparse_vocab_size: int = 50000
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    enable_verification: bool = True
    faithfulness_threshold: float = 0.7
    retrieval_score_threshold: float = 0.3
    reranker_score_threshold: float = 0.5
    citation_support_threshold: float = 0.5
    llm_model: str = "gpt-4o-mini"
    llm_fallback_model: str = ""
    llm_fallback_api_key: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: float = 30.0
    log_level: str = "INFO"
    database_url: str = "sqlite+aiosqlite:///./data/rag.db"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    admin_api_key: str = ""
    redis_url: str = "redis://localhost:6379"
    enable_query_cache: bool = True
    query_cache_ttl: int = 300  # seconds
    enable_rate_limiting: bool = True
    rate_limit_queries: int = 60  # per minute
    rate_limit_ingestion: int = 10  # per minute
    cors_origins: list[str] = ["http://localhost:3000"]


def get_settings() -> Settings:
    return Settings()
