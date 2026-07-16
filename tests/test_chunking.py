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
    # The tail of one chunk should reappear at the head of the next.
    first_tail = chunks[0].split()[-1]
    assert first_tail in chunks[1]


def test_prefers_paragraph_boundaries() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = split_text(text, ChunkConfig(chunk_size=25, chunk_overlap=0))
    assert any("First paragraph." in c for c in chunks)
