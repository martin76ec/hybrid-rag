"""Domain services — pure business logic with no infrastructure dependencies.

Domain services encapsulate cross-cutting domain operations that do not
naturally belong to a single entity or value object.
"""

from __future__ import annotations

from .entities import Chunk, Document


def chunk_text(doc: Document, *, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split a :class:`Document` into overlapping :class:`Chunk` instances.

    Args:
        doc:        The document to chunk.
        chunk_size: Approximate number of characters per chunk.
        overlap:    Number of characters to overlap between consecutive chunks.

    Returns:
        A list of :class:`Chunk` instances preserving the document source.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    index = 0
    for start in range(0, len(doc.text), step):
        end = start + chunk_size
        chunks.append(
            Chunk(text=doc.text[start:end], source=doc.source, chunk_index=index)
        )
        index += 1
    return chunks