from pathlib import Path

from agentic_rag.ingest import discover, ingest, load_chunks

from .conftest import FakeEmbedder, InMemoryStore


def test_discover_finds_markdown_and_text(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("hi", encoding="utf-8")
    (tmp_path / "b.txt").write_text("yo", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")

    found = {p.name for p in discover(tmp_path)}
    assert found == {"a.md", "b.txt"}


def test_discover_on_a_single_file(tmp_path: Path) -> None:
    file = tmp_path / "a.md"
    file.write_text("hi", encoding="utf-8")
    assert discover(file) == [file]


def test_load_chunks_assigns_ids_and_source(tmp_path: Path) -> None:
    file = tmp_path / "doc.md"
    file.write_text("hello world", encoding="utf-8")

    docs = load_chunks([file])
    assert docs[0].id == "doc-0"
    assert docs[0].source == "doc.md"
    assert "hello" in docs[0].text


def test_ingest_embeds_and_stores(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("core hours are 11 to 15 UTC", encoding="utf-8")
    embedder = FakeEmbedder()
    store = InMemoryStore()

    count = ingest(tmp_path, embedder=embedder, store=store)
    assert count >= 1

    results = store.search(embedder.embed(["core hours"])[0], k=1)
    assert results and results[0].source == "a.md"
