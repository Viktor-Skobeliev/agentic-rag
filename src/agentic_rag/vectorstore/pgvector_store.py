"""Postgres + pgvector store. Matches production setups that use pgvector as the
index; run the bundled docker-compose to get a database locally."""

from __future__ import annotations

import re

import psycopg
from pgvector.psycopg import register_vector

from ..models import Document

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PgVectorStore:
    def __init__(self, dsn: str, collection: str, dim: int) -> None:
        if not _IDENT.match(collection):
            raise ValueError(f"invalid collection name: {collection!r}")
        self._dsn = dsn
        self._table = collection
        self._dim = dim
        self._ensure_schema()

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                "id text PRIMARY KEY, text text NOT NULL, source text NOT NULL, "
                f"embedding vector({self._dim}) NOT NULL)"
            )
            conn.commit()

    def add(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        if not docs:
            return
        rows = [
            (doc.id, doc.text, doc.source, emb) for doc, emb in zip(docs, embeddings, strict=True)
        ]
        with self._connect() as conn:
            conn.cursor().executemany(
                f"INSERT INTO {self._table} (id, text, source, embedding) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "text = EXCLUDED.text, source = EXCLUDED.source, embedding = EXCLUDED.embedding",
                rows,
            )
            conn.commit()

    def search(self, embedding: list[float], k: int) -> list[Document]:
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT id, text, source, 1 - (embedding <=> %s) AS score "
                f"FROM {self._table} ORDER BY embedding <=> %s LIMIT %s",
                (embedding, embedding, k),
            )
            return [
                Document(id=row[0], text=row[1], source=row[2], score=float(row[3]))
                for row in cur.fetchall()
            ]

    def reset(self) -> None:
        with self._connect() as conn:
            conn.execute(f"TRUNCATE {self._table}")
            conn.commit()
