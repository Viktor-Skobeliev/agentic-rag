"""Command line entry point: `agentic-rag ingest <path>` and `agentic-rag ask "<question>"`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from .config import Settings, load_settings
from .embeddings import OpenAIEmbedder
from .graph import build_graph
from .ingest import ingest
from .llm import OpenAIRagLLM
from .state import GraphState
from .vectorstore import build_store


def _openai_client(settings: Settings):  # type: ignore[no-untyped-def]
    from openai import OpenAI

    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set (see .env.example).", file=sys.stderr)
        raise SystemExit(2)
    return OpenAI(api_key=settings.openai_api_key)


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> None:
    client = _openai_client(settings)
    embedder = OpenAIEmbedder(client, settings.embedding_model, settings.embedding_dim)
    store = build_store(settings)
    count = ingest(Path(args.path), embedder=embedder, store=store)
    print(f"Ingested {count} chunks from {args.path}.")


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> None:
    client = _openai_client(settings)
    embedder = OpenAIEmbedder(client, settings.embedding_model, settings.embedding_dim)
    store = build_store(settings)
    llm = OpenAIRagLLM(client, settings.chat_model)
    graph = build_graph(
        embedder, store, llm, k=settings.top_k, max_attempts=settings.max_generation_attempts
    )

    result = cast(GraphState, graph.invoke({"question": args.question}))
    print(result.get("answer", ""))
    citations = result.get("citations", [])
    if citations:
        print("\nSources: " + ", ".join(citations))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agentic-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="chunk, embed and load documents")
    p_ingest.add_argument("path", help="file or directory to ingest")

    p_ask = sub.add_parser("ask", help="answer a question from the knowledge base")
    p_ask.add_argument("question", help="the question to answer")

    args = parser.parse_args(argv)
    settings = load_settings()

    if args.command == "ingest":
        _cmd_ingest(args, settings)
    elif args.command == "ask":
        _cmd_ask(args, settings)


if __name__ == "__main__":
    main()
