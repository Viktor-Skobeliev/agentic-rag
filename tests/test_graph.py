from agentic_rag.graph import build_graph
from agentic_rag.models import Document
from agentic_rag.prompts import NO_DATA_MESSAGE

from .conftest import FakeEmbedder, FakeRagLLM, InMemoryStore


def _store_with_docs(embedder: FakeEmbedder) -> InMemoryStore:
    store = InMemoryStore()
    docs = [
        Document(id="remote-0", text="core working hours are 11 to 15 UTC", source="remote.md"),
        Document(id="sec-0", text="production access lasts at most 7 days", source="security.md"),
    ]
    store.add(docs, embedder.embed([d.text for d in docs]))
    return store


def test_happy_path_answers_with_citations() -> None:
    embedder = FakeEmbedder()
    store = _store_with_docs(embedder)
    llm = FakeRagLLM(answer="Core hours are 11 to 15 UTC.", citations=["remote-0"])

    graph = build_graph(embedder, store, llm, k=2)
    result = graph.invoke({"question": "what are the core hours"})

    assert result["answer"] == "Core hours are 11 to 15 UTC."
    assert result["citations"] == ["remote-0"]
    assert result["grounded"] is True


def test_no_relevant_documents_returns_no_data() -> None:
    embedder = FakeEmbedder()
    store = _store_with_docs(embedder)
    # Grader rejects everything -> the graph must not invent an answer.
    llm = FakeRagLLM(relevant_ids=[])

    graph = build_graph(embedder, store, llm, k=2)
    result = graph.invoke({"question": "what is the pet policy"})

    assert result["answer"] == NO_DATA_MESSAGE
    assert result["citations"] == []


def test_regenerates_when_answer_is_not_grounded() -> None:
    embedder = FakeEmbedder()
    store = _store_with_docs(embedder)
    # First grounding check fails, second passes -> exactly two generations.
    llm = FakeRagLLM(answer="Core hours are 11 to 15 UTC.", grounded_sequence=[False, True])

    graph = build_graph(embedder, store, llm, k=2, max_attempts=2)
    result = graph.invoke({"question": "what are the core hours"})

    assert llm.generate_calls == 2
    assert result["grounded"] is True


def test_falls_back_to_no_data_after_max_attempts() -> None:
    embedder = FakeEmbedder()
    store = _store_with_docs(embedder)
    # Grounding never succeeds -> after the attempt cap, honest no-data answer.
    llm = FakeRagLLM(answer="an unsupported claim", grounded_sequence=[False, False, False])

    graph = build_graph(embedder, store, llm, k=2, max_attempts=2)
    result = graph.invoke({"question": "what are the core hours"})

    assert result["answer"] == NO_DATA_MESSAGE
