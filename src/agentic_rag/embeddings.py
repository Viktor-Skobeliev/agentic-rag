"""Embedding providers behind a small interface, so the store and the graph
never depend on a specific vendor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from openai import OpenAI


class Embedder(Protocol):
    """Turns text into vectors. `dim` must match the vector store column."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, client: OpenAI, model: str, dim: int) -> None:
        self._client = client
        self._model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        # The API preserves input order, but sort by index to be safe.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
