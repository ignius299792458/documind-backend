# ============================================================
# Central configuration using pydantic-settings.
# All environment variables are validated and typed here.
# Import `settings` anywhere in the app — never read os.environ directly.
# ============================================================

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All app configuration loaded from environment variables / .env file.
    pydantic-settings automatically reads .env and validates types.
    """

    # --- OpenAI ---
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # --- LangSmith Observability ---
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "documind"

    # --- Cohere Re-ranking ---
    cohere_api_key: str = ""

    # --- Storage Paths ---
    chroma_persist_dir: str = "./chroma_db"
    upload_dir: str = "./uploads"
    chroma_collection_name: str = "documents_docs"

    # --- App Behaviour ---
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"  # comma-separated string from .env
    max_upload_size_mb: int = 50

    # --- splitter: chunking size ---
    chunk_size: int = 1000  # characters per chunk (sweet for context windows of 4k tokens)
    chunk_overlap: int = 200  # characters of overlap between chunks

    # pydantic-settings config: read from .env file, ignore extra vars
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS_ORIGINS into a Python list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        """Convert MB setting to bytes for FastAPI size validation."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.
    Using lru_cache means .env is only read once — not on every request.
    FastAPI's Depends() will call this automatically.
    """
    return Settings()


# Module-level singleton — import this directly in non-FastAPI code
settings = get_settings()
