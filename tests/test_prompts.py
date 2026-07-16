from agentic_rag.models import Document
from agentic_rag.prompts import (
    NO_DATA_MESSAGE,
    format_documents,
    generate_user,
    grade_user,
    grounding_user,
)


def test_format_documents_includes_id_source_and_text() -> None:
    out = format_documents([Document(id="a", text="hello world", source="x.md")])
    assert "a" in out
    assert "x.md" in out
    assert "hello world" in out


def test_no_data_message_is_non_empty() -> None:
    assert NO_DATA_MESSAGE.strip()


def test_user_prompts_carry_the_question_and_answer() -> None:
    docs = [Document(id="a", text="hello", source="x.md")]
    assert "core hours?" in grade_user("core hours?", docs)
    assert "core hours?" in generate_user("core hours?", docs)
    grounding = grounding_user("core hours?", "the answer", docs)
    assert "core hours?" in grounding
    assert "the answer" in grounding
