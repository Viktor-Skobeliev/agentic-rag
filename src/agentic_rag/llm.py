"""The three LLM operations the graph needs, behind one interface.

Keeping them on a single `RagLLM` type means the graph nodes call
`llm.grade(...)` / `llm.generate(...)` / `llm.check_grounding(...)` and never
touch a vendor SDK. Tests supply a fake `RagLLM`; production uses OpenAI.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

from . import prompts
from .models import Document

if TYPE_CHECKING:
    from openai import OpenAI


class RagLLM(Protocol):
    def grade(self, question: str, docs: list[Document]) -> list[str]:
        """Return the ids of documents that actually help answer the question."""

    def generate(self, question: str, docs: list[Document]) -> tuple[str, list[str]]:
        """Return (answer, cited ids) grounded only in `docs`."""

    def check_grounding(self, question: str, answer: str, docs: list[Document]) -> bool:
        """Return True only if every claim in `answer` is supported by `docs`."""


class OpenAIRagLLM:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def _json(self, system: str, user: str) -> dict[str, object]:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}

    def grade(self, question: str, docs: list[Document]) -> list[str]:
        data = self._json(prompts.GRADE_SYSTEM, prompts.grade_user(question, docs))
        raw = data.get("relevant_ids", [])
        items = raw if isinstance(raw, list) else []
        known = {doc.id for doc in docs}
        return [str(i) for i in items if str(i) in known]

    def generate(self, question: str, docs: list[Document]) -> tuple[str, list[str]]:
        data = self._json(prompts.GENERATE_SYSTEM, prompts.generate_user(question, docs))
        answer = str(data.get("answer", prompts.NO_DATA_MESSAGE))
        raw = data.get("citations", [])
        items = raw if isinstance(raw, list) else []
        known = {doc.id for doc in docs}
        cited = [str(c) for c in items if str(c) in known]
        return answer, cited

    def check_grounding(self, question: str, answer: str, docs: list[Document]) -> bool:
        data = self._json(prompts.GROUNDING_SYSTEM, prompts.grounding_user(question, answer, docs))
        return bool(data.get("grounded", False))
