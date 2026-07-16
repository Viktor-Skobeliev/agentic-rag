"""Offline evaluation harness.

Runs the graph over `eval_set.yaml` and reports three things that matter for a
RAG system:

  * retrieval hit-rate  - did we retrieve the document that holds the answer?
  * groundedness         - is the answer supported and does it contain the key fact?
  * no-data correctness  - when the answer is absent, do we correctly say so
                           instead of hallucinating?

Scoring is pure (`score_case`) so it is unit-tested; `main` wires OpenAI and the
configured vector store and runs the whole set.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag.prompts import NO_DATA_MESSAGE  # noqa: E402


@dataclass
class EvalCase:
    question: str
    answerable: bool
    expected_source: str | None = None
    must_contain: list[str] | None = None


@dataclass
class CaseScore:
    question: str
    retrieval_hit: bool | None  # None when not applicable
    grounded_ok: bool | None
    no_data_ok: bool | None

    @property
    def passed(self) -> bool:
        checks = (self.retrieval_hit, self.grounded_ok, self.no_data_ok)
        return all(c for c in checks if c is not None)


def load_cases(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            question=item["question"],
            answerable=item["answerable"],
            expected_source=item.get("expected_source"),
            must_contain=item.get("must_contain"),
        )
        for item in raw
    ]


def score_case(case: EvalCase, answer: str, sources: list[str], grounded: bool) -> CaseScore:
    if not case.answerable:
        return CaseScore(
            question=case.question,
            retrieval_hit=None,
            grounded_ok=None,
            no_data_ok=(answer.strip() == NO_DATA_MESSAGE),
        )

    hit = case.expected_source in sources if case.expected_source else None
    has_facts = all(token in answer for token in (case.must_contain or []))
    return CaseScore(
        question=case.question,
        retrieval_hit=hit,
        grounded_ok=(grounded and has_facts and answer.strip() != NO_DATA_MESSAGE),
        no_data_ok=None,
    )


def _rate(values: list[bool | None]) -> str:
    considered = [v for v in values if v is not None]
    if not considered:
        return "n/a"
    return f"{sum(considered)}/{len(considered)} ({100 * sum(considered) // len(considered)}%)"


def summarize(scores: list[CaseScore]) -> None:
    for s in scores:
        mark = "PASS" if s.passed else "FAIL"
        print(f"[{mark}] {s.question}")
    print("\n--- summary ---")
    print("retrieval hit-rate :", _rate([s.retrieval_hit for s in scores]))
    print("groundedness       :", _rate([s.grounded_ok for s in scores]))
    print("no-data handling   :", _rate([s.no_data_ok for s in scores]))
    passed = sum(s.passed for s in scores)
    print(f"overall            : {passed}/{len(scores)} cases passed")


def main() -> None:
    from agentic_rag.config import load_settings
    from agentic_rag.embeddings import OpenAIEmbedder
    from agentic_rag.graph import build_graph
    from agentic_rag.llm import OpenAIRagLLM
    from agentic_rag.vectorstore import build_store

    settings = load_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set; ingest and eval need a live key.", file=sys.stderr)
        raise SystemExit(2)

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    embedder = OpenAIEmbedder(client, settings.embedding_model, settings.embedding_dim)
    store = build_store(settings)
    llm = OpenAIRagLLM(client, settings.chat_model)
    graph = build_graph(
        embedder, store, llm, k=settings.top_k, max_attempts=settings.max_generation_attempts
    )

    cases = load_cases(Path(__file__).with_name("eval_set.yaml"))
    scores: list[CaseScore] = []
    for case in cases:
        result = graph.invoke({"question": case.question})
        sources = [doc.source for doc in result.get("retrieved", [])]
        scores.append(
            score_case(
                case,
                answer=result.get("answer", ""),
                sources=sources,
                grounded=bool(result.get("grounded", False)),
            )
        )
    summarize(scores)


if __name__ == "__main__":
    main()
