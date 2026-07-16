"""Assemble the agentic-RAG graph.

    retrieve -> grade -> (relevant?) -> generate -> check_grounding
                            |  no                      |  grounded -> END
                            v                          |  not grounded and
                         no_data -> END                |    attempts left -> generate
                                                       |    else -> no_data -> END

The self-check loop is what makes it "agentic": if the answer is not supported
by the retrieved documents, it regenerates until it is (up to a cap), and falls
back to an honest no-data answer instead of shipping an unsupported claim.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from . import nodes
from .embeddings import Embedder
from .llm import RagLLM
from .state import GraphState
from .vectorstore.base import VectorStore

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def _route_after_grade(state: GraphState) -> str:
    return "generate" if state.get("relevant") else "no_data"


def _route_after_check(state: GraphState, *, max_attempts: int) -> str:
    if state.get("grounded"):
        return "end"
    if state.get("attempts", 0) >= max_attempts:
        return "no_data"
    return "generate"


def build_graph(
    embedder: Embedder,
    store: VectorStore,
    llm: RagLLM,
    *,
    k: int = 5,
    max_attempts: int = 2,
) -> CompiledStateGraph[GraphState]:
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", partial(nodes.retrieve, embedder=embedder, store=store, k=k))
    graph.add_node("grade", partial(nodes.grade, llm=llm))
    graph.add_node("generate", partial(nodes.generate, llm=llm))
    graph.add_node("check", partial(nodes.check_grounding, llm=llm))
    graph.add_node("no_data", nodes.answer_no_data)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", _route_after_grade, {"generate": "generate", "no_data": "no_data"}
    )
    graph.add_edge("generate", "check")
    graph.add_conditional_edges(
        "check",
        partial(_route_after_check, max_attempts=max_attempts),
        {"end": END, "generate": "generate", "no_data": "no_data"},
    )
    graph.add_edge("no_data", END)

    return graph.compile()
