"""Application settings loaded from environment / .env (Pydantic v2)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database -----------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://ytkb:ytkb@localhost:5432/ytkb",
        description="Async SQLAlchemy DSN (asyncpg driver).",
    )

    # --- Embeddings ---------------------------------------------------------
    # Fixed at migration 0001. Changing requires a migration + reindex.
    embedding_dim: int = 1536
    embedding_provider: Literal["openai", "voyage", "local", "fake"] = "fake"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: str | None = None
    voyage_api_key: str | None = None

    # --- LLM ----------------------------------------------------------------
    anthropic_api_key: str | None = None
    llm_extraction_model: str = "claude-haiku-4-5-20251001"
    llm_merge_model: str = "claude-sonnet-5"
    llm_chat_model: str = "claude-sonnet-5"

    # --- Feature flags ------------------------------------------------------
    chat_enabled: bool = False

    # --- Chat retrieval -----------------------------------------------------
    chat_topic_k: int = 3
    chat_unit_k: int = 6
    chat_chunk_k: int = 6

    # --- Ingestion ----------------------------------------------------------
    transcript_languages: list[str] = ["it", "en"]
    yt_dlp_cookies_file: str | None = None
    yt_dlp_proxy: str | None = None
    scheduler_enabled: bool = True
    channel_scan_cron: str = "0 3 * * *"  # nightly at 03:00

    # --- Chunking -----------------------------------------------------------
    chunk_target_tokens: int = 500
    chunk_overlap_ratio: float = 0.15

    # --- Knowledge unit extraction -----------------------------------------
    extraction_window_size: int = 4  # chunks per extraction window
    extraction_min_confidence: float = 0.35
    unit_near_dup_threshold: float = 0.93  # cosine similarity to treat as dup

    # --- Topic assignment thresholds ---------------------------------------
    theta_high: float = 0.82
    theta_low: float = 0.62
    topic_promote_after_units: int = 5
    topic_similarity_review_threshold: float = 0.9  # centroid sim to propose merge

    # --- ASR fallback -------------------------------------------------------
    asr_provider: Literal["faster_whisper", "assemblyai", "openai", "none"] = "none"
    asr_whisper_model: str = "small"
    asr_max_duration_s: int = 1800

    # --- Cost estimate (USD per 1M tokens; rough, for budget reports) -------
    price_embedding_per_mtok: float = 0.02
    price_extraction_per_mtok: float = 0.80  # Haiku input
    price_merge_per_mtok: float = 3.0  # Sonnet input

    @property
    def sync_database_url(self) -> str:
        """Sync DSN (psycopg) for Alembic / tooling."""
        return self.database_url.replace("+asyncpg", "+psycopg")


@lru_cache
def get_settings() -> Settings:
    return Settings()
