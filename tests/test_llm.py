from typing import Any, cast

from agentic_rag.llm import OpenAIRagLLM
from agentic_rag.models import Document
from agentic_rag.prompts import NO_DATA_MESSAGE

from .openai_fakes import FakeChatClient

DOCS = [Document(id="a", text="core hours 11 to 15", source="remote.md")]


def _llm(content: str) -> OpenAIRagLLM:
    return OpenAIRagLLM(cast(Any, FakeChatClient([content])), model="m")


def test_grade_keeps_only_known_ids() -> None:
    result = _llm('{"relevant_ids": ["a", "does-not-exist"]}').grade("q", DOCS)
    assert result == ["a"]


def test_grade_handles_non_list() -> None:
    assert _llm('{"relevant_ids": "a"}').grade("q", DOCS) == []


def test_generate_returns_answer_and_filtered_citations() -> None:
    answer, citations = _llm(
        '{"answer": "Core hours are 11 to 15.", "citations": ["a", "x"]}'
    ).generate("q", DOCS)
    assert answer == "Core hours are 11 to 15."
    assert citations == ["a"]


def test_generate_defaults_to_no_data_when_answer_missing() -> None:
    answer, citations = _llm("{}").generate("q", DOCS)
    assert answer == NO_DATA_MESSAGE
    assert citations == []


def test_check_grounding_parses_boolean() -> None:
    assert _llm('{"grounded": true}').check_grounding("q", "a", DOCS) is True
    assert _llm('{"grounded": false}').check_grounding("q", "a", DOCS) is False


def test_malformed_json_is_handled_as_empty() -> None:
    # A non-JSON body must not crash the pipeline; it degrades to "no relevant".
    assert _llm("not json at all").grade("q", DOCS) == []
