"""Application configuration loaded from environment variables / .env file."""

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> repository root is three levels up from `core`.
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values are read from the environment first and fall back to the root
    ``.env`` file. Credentials are never hardcoded; ``db_user`` and
    ``db_password`` are required so a misconfigured environment fails fast
    with a clear validation error.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "college_db"
    db_user: str
    db_password: str
    db_echo: bool = False

    # Root directory for generated RAG artifacts (metadata + knowledge docs).
    rag_output_dir: Path = REPO_ROOT / "rag"

    # Phase 3: local embedding model + FAISS index location.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_index_dir: Path = REPO_ROOT / "rag" / "index"
    embedding_batch_size: int = 32

    # Phase 4/4.1: LLM provider for Text-to-SQL + retrieval/context budgets.
    # Default provider is Google Gemini (cloud); Ollama remains available as
    # a local fallback via LLM_PROVIDER=ollama. The Gemini API key is a
    # secret (SecretStr) and is never hardcoded, logged, or printed.
    llm_provider: str = "gemini"
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.6-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    llm_timeout_seconds: float = 120.0
    rag_top_k: int = 16
    max_context_chars: int = 6000

    # Phase 5: read-only SQL execution.
    # Execution uses a dedicated low-privilege role (SELECT-only on the six
    # tables). When ``exec_db_user`` is empty, execution stays disabled and
    # generation-only endpoints keep working unchanged.
    sql_max_rows: int = 500
    sql_statement_timeout_ms: int = 5000
    exec_db_user: str = ""
    exec_db_password: SecretStr = SecretStr("")

    @property
    def active_llm_model(self) -> str:
        """Model name belonging to the configured ``llm_provider``."""
        return self.gemini_model if self.llm_provider == "gemini" else self.ollama_model

    @property
    def database_url(self) -> str:
        """Asynchronous-driver-free SQLAlchemy URL for psycopg 3."""
        user = quote_plus(self.db_user)
        password = quote_plus(self.db_password)
        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def execution_enabled(self) -> bool:
        """True when dedicated read-only execution credentials are configured."""
        return bool(self.exec_db_user.strip()) and bool(
            self.exec_db_password.get_secret_value().strip()
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance (safe to call from anywhere)."""
    return Settings()
