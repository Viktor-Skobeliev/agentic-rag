from agentic_rag.models import Document
from agentic_rag.nodes import answer_no_data, check_grounding, generate, grade, retrieve
from agentic_rag.prompts import NO_DATA_MESSAGE

from .conftest import FakeEmbedder, FakeRagLLM, InMemoryStore

DOCS = [
    Document(id="a", text="core hours are 11 to 15 UTC", source="remote.md"),
    Document(id="b", text="production access lasts 7 days", source="security.md"),
]


def test_retrieve_pulls_from_store() -> None:
    embedder = FakeEmbedder()
    store = InMemoryStore()
    store.add(DOCS, embedder.embed([d.text for d in DOCS]))

    out = retrieve({"question": "core hours"}, embedder=embedder, store=store, k=1)
    assert out["retrieved"][0].id == "a"


def test_grade_keeps_only_relevant() -> None:
    out = grade({"question": "q", "retrieved": DOCS}, llm=FakeRagLLM(relevant_ids=["a"]))
    assert [d.id for d in out["relevant"]] == ["a"]


def test_grade_with_no_documents() -> None:
    out = grade({"question": "q", "retrieved": []}, llm=FakeRagLLM())
    assert out["relevant"] == []


def test_generate_sets_answer_and_increments_attempts() -> None:
    llm = FakeRagLLM(answer="Core hours are 11 to 15.", citations=["a"])
    out = generate({"question": "q", "relevant": DOCS, "attempts": 0}, llm=llm)
    assert out["answer"] == "Core hours are 11 to 15."
    assert out["citations"] == ["a"]
    assert out["attempts"] == 1


def test_check_grounding_reports_llm_verdict() -> None:
    out = check_grounding(
        {"question": "q", "answer": "a", "relevant": DOCS},
        llm=FakeRagLLM(grounded_sequence=[True]),
    )
    assert out["grounded"] is True


def test_no_data_node_returns_the_honest_message() -> None:
    out = answer_no_data({})
    assert out["answer"] == NO_DATA_MESSAGE
    assert out["citations"] == []
