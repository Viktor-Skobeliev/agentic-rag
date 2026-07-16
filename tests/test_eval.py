from pathlib import Path

from agentic_rag.prompts import NO_DATA_MESSAGE
from run_eval import EvalCase, load_cases, score_case


def test_answerable_case_scored_by_retrieval_and_facts() -> None:
    case = EvalCase(
        question="core hours?",
        answerable=True,
        expected_source="remote-work.md",
        must_contain=["11:00", "15:00"],
    )
    score = score_case(
        case,
        answer="Core hours are 11:00 to 15:00 UTC.",
        sources=["remote-work.md", "security.md"],
        grounded=True,
    )
    assert score.retrieval_hit is True
    assert score.grounded_ok is True
    assert score.no_data_ok is None
    assert score.passed


def test_missing_key_fact_fails_groundedness() -> None:
    case = EvalCase(
        question="core hours?",
        answerable=True,
        expected_source="remote-work.md",
        must_contain=["11:00"],
    )
    score = score_case(case, answer="Some vague answer.", sources=["remote-work.md"], grounded=True)
    assert score.grounded_ok is False
    assert not score.passed


def test_unanswerable_case_wants_no_data() -> None:
    case = EvalCase(question="pet policy?", answerable=False)
    good = score_case(case, answer=NO_DATA_MESSAGE, sources=[], grounded=True)
    bad = score_case(case, answer="Pets are welcome.", sources=[], grounded=True)
    assert good.no_data_ok is True and good.passed
    assert bad.no_data_ok is False and not bad.passed


def test_load_cases_reads_the_yaml() -> None:
    cases = load_cases(Path(__file__).resolve().parents[1] / "eval" / "eval_set.yaml")
    assert len(cases) >= 6
    assert any(not c.answerable for c in cases)
