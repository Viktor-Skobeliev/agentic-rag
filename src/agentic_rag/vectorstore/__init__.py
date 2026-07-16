"""Vector store selection. `build_store` returns a backend by name so callers
depend only on the `VectorStore` protocol."""

from __future__ import annotations

from ..config import Settings
from .base import VectorStore

__all__ = ["VectorStore", "build_store"]


def build_store(settings: Settings) -> VectorStore:
    backend = settings.vector_backend.lower()
    if backend == "chroma":
        from .chroma_store import ChromaStore

        return ChromaStore(path=settings.chroma_path, collection=settings.collection)
    if backend == "pgvector":
        from .pgvector_store import PgVectorStore

        return PgVectorStore(
            dsn=settings.pg_dsn, collection=settings.collection, dim=settings.embedding_dim
        )
    raise ValueError(f"unknown vector backend: {settings.vector_backend!r}")
