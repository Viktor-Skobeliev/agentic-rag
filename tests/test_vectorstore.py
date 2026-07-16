import pytest

from agentic_rag.models import Document
from agentic_rag.vectorstore.chroma_store import ChromaStore
from agentic_rag.vectorstore.pgvector_store import PgVectorStore

from .conftest import FakeEmbedder


def test_chroma_returns_nearest_document(tmp_path) -> None:  # type: ignore[no-untyped-def]
    embedder = FakeEmbedder()
    store = ChromaStore(path=str(tmp_path / "chroma"), collection="test")

    docs = [
        Document(id="a", text="core working hours are 11 to 15 UTC", source="remote.md"),
        Document(id="b", text="the meal allowance while travelling is 50", source="expenses.md"),
    ]
    store.add(docs, embedder.embed([d.text for d in docs]))

    query = embedder.embed(["what are the core working hours"])[0]
    results = store.search(query, k=1)

    assert len(results) == 1
    assert results[0].id == "a"
    assert results[0].score is not None


def test_pgvector_rejects_unsafe_collection_name() -> None:
    # Validation happens before any DB connection, so this needs no database.
    with pytest.raises(ValueError):
        PgVectorStore(dsn="postgresql://localhost/x", collection="drop table", dim=8)
