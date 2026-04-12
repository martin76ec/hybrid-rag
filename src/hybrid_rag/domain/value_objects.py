"""Domain value objects.

Value objects are immutable, compared by value, and carry no identity of their
own.  They describe *things* in the domain rather than *who* they are.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingVector:
    """A dense vector produced by an embedding model.

    Attributes:
        values: The float components of the embedding.
    """

    values: list[float]


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata associated with a single vector in the store.

    Attributes:
        source:      File path or URI of the original document.
        chunk_index: Position of the chunk in the original document.
        text:        The raw text of the chunk.
    """

    source: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    """A single result returned by a similarity search.

    Attributes:
        chunk: The chunk metadata that matched.
        score: Distance/similarity score (lower is better for L2 distance).
    """

    chunk: ChunkMetadata
    score: float