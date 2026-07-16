from typing import Any, cast

from agentic_rag.embeddings import OpenAIEmbedder

from .openai_fakes import FakeEmbeddingsClient


def test_empty_input_makes_no_api_call() -> None:
    embedder = OpenAIEmbedder(cast(Any, FakeEmbeddingsClient()), model="m", dim=4)
    assert embedder.embed([]) == []


def test_restores_input_order_by_index() -> None:
    # The fake returns vectors reversed; the embedder must re-sort by index.
    embedder = OpenAIEmbedder(cast(Any, FakeEmbeddingsClient(reverse=True)), model="m", dim=2)
    vectors = embedder.embed(["a", "bb"])
    assert len(vectors) == 2
    assert vectors[0] == [0.0, 1.0]  # index 0, len("a") == 1
    assert vectors[1] == [1.0, 2.0]  # index 1, len("bb") == 2


def test_dim_is_exposed() -> None:
    embedder = OpenAIEmbedder(cast(Any, FakeEmbeddingsClient()), model="m", dim=1536)
    assert embedder.dim == 1536
