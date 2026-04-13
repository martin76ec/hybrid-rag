"""Domain services — pure business logic with no infrastructure dependencies.

Domain services encapsulate cross-cutting domain operations that do not
naturally belong to a single entity or value object.
"""

from __future__ import annotations

from .entities import Chunk, Document
from .value_objects import ChunkMetadata, RetrievalResult


def chunk_text(
    doc: Document, *, chunk_size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """Split a :class:`Document` into overlapping :class:`Chunk` instances.

    Chunk boundaries are snapped to the nearest whitespace so that words
    are never split across chunks.

    Args:
        doc:        The document to chunk.
        chunk_size: Approximate number of characters per chunk.
        overlap:    Number of characters to overlap between consecutive chunks.

    Returns:
        A list of :class:`Chunk` instances preserving the document source.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
    step = chunk_size - overlap
    text = doc.text
    chunks: list[Chunk] = []
    index = 0
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            snap = end
            while snap > start and text[snap - 1] not in (" ", "\n", "\t"):
                snap -= 1
            if snap > start:
                end = snap
        chunks.append(Chunk(text=text[start:end], source=doc.source, chunk_index=index))
        index += 1
        next_start = start + step
        if next_start < len(text) and next_start > 0:
            snap = next_start
            while snap < len(text) and text[snap] not in (" ", "\n", "\t"):
                snap += 1
            if snap < len(text):
                next_start = snap + 1
        if next_start <= start:
            next_start = start + step
        start = next_start
    return chunks


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Each result's RRF score is ``sum(1 / (k + rank))`` across all lists
    where it appears.  Results are returned sorted by descending RRF score.

    Chunk identity is determined by ``(source, chunk_index)``.

    Args:
    ranked_lists: One or more ranked lists of retrieval results.
    k:           RRF smoothing constant (default 60).

    Returns:
    A single fused and re-ranked list of :class:`RetrievalResult`.
    """
    scores: dict[tuple[str, int], float] = {}
    chunk_map: dict[tuple[str, int], ChunkMetadata] = {}

    for results in ranked_lists:
        for rank, rr in enumerate(results, start=1):
            key = (rr.chunk.source, rr.chunk.chunk_index)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            chunk_map[key] = rr.chunk

    fused = [
        RetrievalResult(chunk=chunk_map[key], score=score)
        for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return fused
