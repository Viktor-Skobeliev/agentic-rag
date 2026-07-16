from pathlib import Path

import pytest

from agentic_rag.config import Settings
from agentic_rag.vectorstore import build_store


def test_build_store_returns_chroma_backend(tmp_path: Path) -> None:
    settings = Settings(
        vector_backend="chroma",
        chroma_path=str(tmp_path / "chroma"),
        collection="knowledge_base",
    )
    store = build_store(settings)
    assert hasattr(store, "add")
    assert hasattr(store, "search")
    assert hasattr(store, "reset")


def test_build_store_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        build_store(Settings(vector_backend="nope"))
