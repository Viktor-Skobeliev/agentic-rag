"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    """A retrievable unit of text.

    `id` is stable across ingestion runs so the same chunk keeps its identity.
    `score` is the similarity to a query and is only set on retrieved documents.
    """

    id: str
    text: str
    source: str
    score: float | None = None
