from agentic_rag.chunking import ChunkConfig, split_text


def test_empty_text_yields_no_chunks() -> None:
    assert split_text("   ") == []


def test_short_text_is_a_single_chunk() -> None:
    assert split_text("A short note.") == ["A short note."]


def test_chunks_respect_size() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, ChunkConfig(chunk_size=200, chunk_overlap=40))
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunks_overlap_to_preserve_context() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, ChunkConfig(chunk_size=200, chunk_overlap=40))
    first_tail = chunks[0].split()[-1]
    assert first_tail in chunks[1]


def test_prefers_paragraph_boundaries() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = split_text(text, ChunkConfig(chunk_size=25, chunk_overlap=0))
    assert any("First paragraph." in c for c in chunks)


def test_text_shorter_than_chunk_size_is_not_split() -> None:
    text = "one two three four five"
    assert split_text(text, ChunkConfig(chunk_size=100)) == [text]


def test_a_single_oversized_token_is_hard_split() -> None:
    text = "x" * 500  # no separators to break on
    chunks = split_text(text, ChunkConfig(chunk_size=100, chunk_overlap=10))
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_unicode_is_preserved() -> None:
    text = "Робочі години з 11 до 15 за Києвом. " * 20
    chunks = split_text(text, ChunkConfig(chunk_size=120, chunk_overlap=20))
    assert chunks
    assert "Києвом" in " ".join(chunks)
