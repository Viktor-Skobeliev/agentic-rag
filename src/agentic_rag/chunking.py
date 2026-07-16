"""Split documents into overlapping chunks for embedding.

A small recursive splitter: it breaks on the largest natural boundary first
(paragraph, then line, then sentence, then word) so a chunk rarely cuts through
the middle of a sentence. Adjacent pieces are then merged up to `chunk_size`
with a `chunk_overlap` tail carried into the next chunk to preserve context.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Break `text` into pieces no larger than `chunk_size` where possible."""
    if len(text) <= chunk_size or not separators:
        return [text]

    sep, *rest = separators
    pieces = text.split(sep) if sep else list(text)

    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) <= chunk_size:
            out.append(piece)
        else:
            out.extend(_recursive_split(piece, rest, chunk_size))
    return out


def _merge(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily merge small pieces into chunks, carrying an overlap tail."""
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        tail = current[-overlap:].strip() if overlap and current else ""
        current = f"{tail} {piece}".strip() if tail else piece

        # A single piece longer than chunk_size is hard-split as a last resort.
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - overlap :]

    if current:
        chunks.append(current)
    return chunks


def split_text(text: str, config: ChunkConfig | None = None) -> list[str]:
    """Split `text` into overlapping chunks."""
    cfg = config or ChunkConfig()
    if not text.strip():
        return []
    pieces = _recursive_split(text, DEFAULT_SEPARATORS, cfg.chunk_size)
    return _merge(pieces, cfg.chunk_size, cfg.chunk_overlap)
