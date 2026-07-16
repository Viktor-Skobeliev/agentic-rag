"""Test doubles that let the whole pipeline run offline, without any API calls.

FakeEmbedder turns text into a deterministic bag-of-words vector, so documents
that share words with the query rank higher — enough for retrieval to be
meaningful in tests. InMemoryStore is a tiny cosine-similarity store.
FakeRagLLM returns scripted grading / generation / grounding results.
"""

from __future__ import annotations

import hashlib
import math

from agentic_rag.models import Document


class FakeEmbedder:
    dim = 64

    def _hash(self, word: str) -> int:
        digest = hashlib.md5(word.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in text.lower().split():
                vec[self._hash(word)] += 1.0
            vectors.append(vec)
        return vectors


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryStore:
    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._embeddings: list[list[float]] = []

    def add(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        self._docs.extend(docs)
        self._embeddings.extend(embeddings)

    def search(self, embedding: list[float], k: int) -> list[Document]:
        scored = [
            Document(id=doc.id, text=doc.text, source=doc.source, score=_cosine(embedding, emb))
            for doc, emb in zip(self._docs, self._embeddings, strict=True)
        ]
        scored.sort(key=lambda d: d.score or 0.0, reverse=True)
        return scored[:k]

    def reset(self) -> None:
        self._docs.clear()
        self._embeddings.clear()


class FakeRagLLM:
    """Scripted LLM. `grounded_sequence` lets a test drive the self-check loop."""

    def __init__(
        self,
        *,
        relevant_ids: list[str] | None = None,
        answer: str = "the answer",
        citations: list[str] | None = None,
        grounded_sequence: list[bool] | None = None,
    ) -> None:
        self._relevant_ids = relevant_ids
        self._answer = answer
        self._citations = citations
        self._grounded_sequence = list(grounded_sequence or [])
        self.generate_calls = 0

    def grade(self, question: str, docs: list[Document]) -> list[str]:
        if self._relevant_ids is not None:
            return self._relevant_ids
        return [doc.id for doc in docs]

    def generate(self, question: str, docs: list[Document]) -> tuple[str, list[str]]:
        self.generate_calls += 1
        citations = self._citations if self._citations is not None else [d.id for d in docs]
        return self._answer, citations

    def check_grounding(self, question: str, answer: str, docs: list[Document]) -> bool:
        if self._grounded_sequence:
            return self._grounded_sequence.pop(0)
        return True
