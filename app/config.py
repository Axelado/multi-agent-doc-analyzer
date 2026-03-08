"""Configuration centralisée de l'application."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- PostgreSQL ---
    postgres_user: str = "nlp_user"
    postgres_password: str = "nlp_secret_password"
    postgres_db: str = "nlp_platform"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_num_ctx: int = 6144
    ollama_num_predict: int = 3072

    # --- Modèles HuggingFace (locaux) ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-base"
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    embedding_device: str = "cpu"  # auto | cpu | cuda
    reranker_device: str = "cpu"   # auto | cpu | cuda
    nli_device: str = "cpu"        # auto | cpu | cuda

    # --- Chemins ---
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 100

    # --- Langfuse (optionnel) ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # --- Application ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    dashboard_port: int = 8501
    log_level: str = "INFO"

    # --- RAG ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    nli_threshold: float = 0.7
    analyst_context_max_chars: int = 14000
    analyst_summary_target_words: int = 300
    analyst_summary_min_words: int = 180
    analyst_summary_retry_count: int = 1

    # --- Fiabilité pipeline ---
    step_timeout_parse_sec: int = 120
    step_timeout_index_sec: int = 240
    step_timeout_analyze_sec: int = 300
    step_timeout_verify_sec: int = 300
    step_timeout_edit_sec: int = 180

    step_retry_parse: int = 2
    step_retry_index: int = 2
    step_retry_analyze: int = 2
    step_retry_verify: int = 2
    step_retry_edit: int = 2

    step_backoff_base_sec: int = 5
    step_backoff_max_sec: int = 60

    def ensure_dirs(self) -> None:
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
