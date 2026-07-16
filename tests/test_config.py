from agentic_rag.config import Settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.vector_backend == "chroma"
    assert settings.top_k == 5
    assert settings.embedding_dim == 1536
    assert settings.max_generation_attempts == 2


def test_field_override() -> None:
    assert Settings(vector_backend="pgvector").vector_backend == "pgvector"
