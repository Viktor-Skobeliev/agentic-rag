"""The state passed between LangGraph nodes.

`total=False` because each node fills in only the keys it produces; LangGraph
merges the partial updates into the running state.
"""

from __future__ import annotations

from typing import TypedDict

from .models import Document


class GraphState(TypedDict, total=False):
    question: str
    retrieved: list[Document]
    relevant: list[Document]
    answer: str
    citations: list[str]
    grounded: bool
    attempts: int
