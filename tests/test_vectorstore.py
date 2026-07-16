from pathlib import Path

import pytest

from agentic_rag.models import Document
from agentic_rag.vectorstore.chroma_store import ChromaStore
from agentic_rag.vectorstore.pgvector_store import PgVectorStore

from .conftest import FakeEmbedder

DOCS = [
    Document(id="a", text="core working hours are 11 to 15 UTC", source="remote.md"),
    Document(id="b", text="the meal allowance while travelling is 50", source="expenses.md"),
]


def _store(tmp_path: Path, name: str = "test") -> ChromaStore:
    return ChromaStore(path=str(tmp_path / "chroma"), collection=name)


def test_chroma_returns_nearest_document(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = _store(tmp_path)
    store.add(DOCS, embedder.embed([d.text for d in DOCS]))

    results = store.search(embedder.embed(["what are the core working hours"])[0], k=1)
    assert len(results) == 1
    assert results[0].id == "a"
    assert results[0].score is not None


def test_chroma_limits_results_to_k(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = _store(tmp_path)
    store.add(DOCS, embedder.embed([d.text for d in DOCS]))
    assert len(store.search(embedder.embed(["hours"])[0], k=1)) == 1


def test_chroma_upsert_overwrites_by_id(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = _store(tmp_path)
    store.add([Document(id="a", text="old text", source="s.md")], embedder.embed(["old text"]))
    store.add([Document(id="a", text="new text", source="s.md")], embedder.embed(["new text"]))

    results = store.search(embedder.embed(["new text"])[0], k=5)
    assert len([r for r in results if r.id == "a"]) == 1
    assert results[0].text == "new text"


def test_chroma_reset_clears_everything(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = _store(tmp_path)
    store.add(DOCS, embedder.embed([d.text for d in DOCS]))
    store.reset()
    assert store.search(embedder.embed(["hours"])[0], k=5) == []


def test_chroma_empty_add_is_a_noop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([], [])
    assert store.search([0.0] * FakeEmbedder.dim, k=5) == []


def test_pgvector_rejects_unsafe_collection_name() -> None:
    # Validation happens before any DB connection, so this needs no database.
    with pytest.raises(ValueError):
        PgVectorStore(dsn="postgresql://localhost/x", collection="drop table", dim=8)
