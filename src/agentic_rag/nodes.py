"""LangGraph nodes. Each takes the running state plus its dependencies and
returns a partial state update. Dependencies are bound in `graph.build_graph`,
which keeps the nodes pure and easy to test.
"""

from __future__ import annotations

from .embeddings import Embedder
from .llm import RagLLM
from .prompts import NO_DATA_MESSAGE
from .state import GraphState
from .vectorstore.base import VectorStore


def retrieve(state: GraphState, *, embedder: Embedder, store: VectorStore, k: int) -> GraphState:
    embedding = embedder.embed([state["question"]])[0]
    return {"retrieved": store.search(embedding, k)}


def grade(state: GraphState, *, llm: RagLLM) -> GraphState:
    docs = state.get("retrieved", [])
    if not docs:
        return {"relevant": []}
    keep = set(llm.grade(state["question"], docs))
    return {"relevant": [doc for doc in docs if doc.id in keep]}


def generate(state: GraphState, *, llm: RagLLM) -> GraphState:
    answer, citations = llm.generate(state["question"], state.get("relevant", []))
    return {
        "answer": answer,
        "citations": citations,
        "attempts": state.get("attempts", 0) + 1,
    }


def check_grounding(state: GraphState, *, llm: RagLLM) -> GraphState:
    grounded = llm.check_grounding(
        state["question"], state.get("answer", ""), state.get("relevant", [])
    )
    return {"grounded": grounded}


def answer_no_data(state: GraphState) -> GraphState:
    # A wrong confident answer is worse than an honest "I don't know".
    return {"answer": NO_DATA_MESSAGE, "citations": [], "grounded": True}
