"""Ingestion: read documents, chunk them, embed, and load the vector store."""

from __future__ import annotations

from pathlib import Path

from .chunking import ChunkConfig, split_text
from .embeddings import Embedder
from .models import Document
from .vectorstore.base import VectorStore


def load_chunks(paths: list[Path], config: ChunkConfig | None = None) -> list[Document]:
    """Read each file and split it into identified chunks."""
    docs: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(split_text(text, config)):
            docs.append(Document(id=f"{path.stem}-{index}", text=chunk, source=path.name))
    return docs


def discover(root: Path) -> list[Path]:
    """Find markdown/text files under `root` (or return `root` itself if a file)."""
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in {".md", ".txt"})


def ingest(
    root: Path,
    *,
    embedder: Embedder,
    store: VectorStore,
    config: ChunkConfig | None = None,
    reset: bool = True,
) -> int:
    """Ingest everything under `root`. Returns the number of chunks stored."""
    docs = load_chunks(discover(root), config)
    if not docs:
        return 0
    if reset:
        store.reset()
    embeddings = embedder.embed([doc.text for doc in docs])
    store.add(docs, embeddings)
    return len(docs)
