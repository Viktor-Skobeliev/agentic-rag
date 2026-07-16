from agentic_rag.graph import _route_after_check, _route_after_grade
from agentic_rag.models import Document
from agentic_rag.state import GraphState


def test_route_to_generate_when_relevant_docs_exist() -> None:
    state: GraphState = {"relevant": [Document(id="a", text="t", source="s")]}
    assert _route_after_grade(state) == "generate"


def test_route_to_no_data_when_nothing_relevant() -> None:
    empty: GraphState = {"relevant": []}
    assert _route_after_grade(empty) == "no_data"
    assert _route_after_grade({}) == "no_data"


def test_grounded_answer_ends() -> None:
    state: GraphState = {"grounded": True}
    assert _route_after_check(state, max_attempts=2) == "end"


def test_ungrounded_with_attempts_left_regenerates() -> None:
    state: GraphState = {"grounded": False, "attempts": 1}
    assert _route_after_check(state, max_attempts=2) == "generate"


def test_ungrounded_at_cap_falls_back_to_no_data() -> None:
    state: GraphState = {"grounded": False, "attempts": 2}
    assert _route_after_check(state, max_attempts=2) == "no_data"
