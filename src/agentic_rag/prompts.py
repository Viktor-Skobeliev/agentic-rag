"""Prompt templates for the three LLM steps: grading, generation, grounding check.

Kept in one place so they are easy to review and version. Each asks for JSON so
the output is parseable and testable.
"""

from __future__ import annotations

from .models import Document

NO_DATA_MESSAGE = "I could not find an answer to this in the knowledge base."


def format_documents(docs: list[Document]) -> str:
    return "\n\n".join(f"[{doc.id}] (source: {doc.source})\n{doc.text}" for doc in docs)


GRADE_SYSTEM = (
    "You are a strict relevance grader for a retrieval system. "
    "Given a question and candidate documents, return only the ids of documents "
    "that actually help answer the question. If none help, return an empty list. "
    'Respond as JSON: {"relevant_ids": ["id1", "id2"]}.'
)


def grade_user(question: str, docs: list[Document]) -> str:
    return f"Question:\n{question}\n\nCandidate documents:\n{format_documents(docs)}"


GENERATE_SYSTEM = (
    "You answer strictly from the provided documents. "
    "Do not use outside knowledge. If the documents do not contain the answer, "
    f'say exactly: "{NO_DATA_MESSAGE}". '
    "Cite the document ids you used. "
    'Respond as JSON: {"answer": "...", "citations": ["id1"]}.'
)


def generate_user(question: str, docs: list[Document]) -> str:
    return f"Question:\n{question}\n\nDocuments:\n{format_documents(docs)}"


GROUNDING_SYSTEM = (
    "You check whether an answer is fully supported by the provided documents. "
    "Return grounded=true only if every claim in the answer is backed by the documents. "
    'Respond as JSON: {"grounded": true|false}.'
)


def grounding_user(question: str, answer: str, docs: list[Document]) -> str:
    return f"Question:\n{question}\n\nAnswer:\n{answer}\n\nDocuments:\n{format_documents(docs)}"
