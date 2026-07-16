"""Chroma-backed vector store. Zero infrastructure: it persists to a local
directory, so the repo runs after a plain `git clone`."""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..models import Document


class ChromaStore:
    def __init__(self, path: str, collection: str) -> None:
        self._client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._name = collection
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        if not docs:
            return
        self._collection.upsert(
            ids=[doc.id for doc in docs],
            documents=[doc.text for doc in docs],
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=[{"source": doc.source} for doc in docs],
        )

    def search(self, embedding: list[float], k: int) -> list[Document]:
        result = self._collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result["ids"] or [[]])[0]
        texts = (result["documents"] or [[]])[0]
        metas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]

        docs: list[Document] = []
        for doc_id, text, meta, distance in zip(ids, texts, metas, distances, strict=True):
            docs.append(
                Document(
                    id=doc_id,
                    text=text,
                    source=str(meta.get("source", "")),
                    score=1.0 - float(distance),  # cosine distance -> similarity
                )
            )
        return docs

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )
