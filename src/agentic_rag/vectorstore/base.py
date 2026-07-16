"""The vector store interface the rest of the app depends on.

Two backends implement it (Chroma, pgvector). Swapping them is a config change,
not a code change — the graph never imports a concrete store.
"""

from __future__ import annotations

from typing import Protocol

from ..models import Document


class VectorStore(Protocol):
    def add(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        """Insert documents with their embeddings (upsert by id)."""

    def search(self, embedding: list[float], k: int) -> list[Document]:
        """Return the k nearest documents, each with a similarity `score`."""

    def reset(self) -> None:
        """Drop all stored documents (used before a fresh ingest)."""
