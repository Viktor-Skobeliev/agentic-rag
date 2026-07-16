"""Runtime settings, read from environment / .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    embedding_dim: int = 1536  # text-embedding-3-small

    # "chroma" (zero-setup, default) or "pgvector"
    vector_backend: str = "chroma"
    chroma_path: str = ".chroma"
    collection: str = "knowledge_base"
    pg_dsn: str = "postgresql://postgres:postgres@localhost:5432/rag"

    top_k: int = 5
    max_generation_attempts: int = 2


def load_settings() -> Settings:
    return Settings()
