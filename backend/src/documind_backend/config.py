# ============================================================
# Central configuration using pydantic-settings.
# All environment variables are validated and typed here.
# Import `settings` anywhere in the app — never read os.environ directly.
# ============================================================

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor all relative paths to the backend/ directory so temp files,
# uploads, and .env resolution work regardless of the CWD the server
# is launched from (e.g. parent documind/ or backend/ itself).
# config.py lives at backend/src/documind_backend/config.py
#   → .parent    = backend/src/documind_backend/
#   → .parent x2 = backend/src/
#   → .parent x3 = backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    All app configuration loaded from environment variables / .env file.
    pydantic-settings automatically reads .env and validates types.
    """

    # --- ollama ---
    ollama_embedding_model: str = "nomic-embed-text:latest"
    ollama_chat_model: str = "llama3.2:latest"
    ollama_creativity_temperature: float = 0.5 # 0.0 = deterministic (less creative) ----> 1.0 = creative (more random)
    ollama_base_url: str = "http://localhost:11434"

    # --- LangSmith Observability ---
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "documind"

    # --- Retrieval tuning ---
    retrieval_top_k: int = 10  # fetch top 10 before re-ranking
    rerank_top_n: int = 3  # keep top 3 after re-ranking
    confidence_threshold: float = 0.015  # drop chunks below this score
    use_reranking: bool = True  # set False to skip re-ranking entirely

    # --- Chroma Storage Config ---
    upload_dir: str = str(BACKEND_DIR / "uploads")
    chroma_collection_name: str = "documents_docs"
    chroma_host: str = "127.0.0.1"
    chroma_port: int = 8001
    chroma_use_http: bool = True

    # --- App Behaviour ---
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000,http://192.168.1.9:3000"  # comma-separated string from .env
    max_upload_size_mb: int = 50

    # --- splitter: chunking size ---
    chunk_size: int = 1000  # characters per chunk (sweet for context windows of 4k tokens)
    chunk_overlap: int = 200  # characters of overlap between chunks

    # pydantic-settings config: .env resolved relative to backend/ dir
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
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


# @lru_cache() ( commenting lru_cache, for testing purpose: setting value at runtime)
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.
    Using lru_cache means .env is only read once — not on every request.
    FastAPI's Depends() will call this automatically.
    """
    return Settings()


# Module-level singleton — import this directly in non-FastAPI code
settings = get_settings()
