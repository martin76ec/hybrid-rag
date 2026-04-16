"""Domain value objects.

Value objects are immutable, compared by value, and carry no identity of their
own.  They describe *things* in the domain rather than *who* they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
class Triple:
    """A knowledge-graph triple (subject, predicate, object).

    Attributes:
        subject:      The source entity.
        predicate:    The relationship label.
        obj:          The target entity or value.
        source:       The document source this triple was extracted from.
        chunk_index:  The chunk index within the source document.
        chunk_text:   The text of the chunk this triple was extracted from.
    """

    subject: str
    predicate: str
    obj: str
    source: str = ""
    chunk_index: int = 0
    chunk_text: str = ""


@dataclass(frozen=True)
class DocumentAnalysis:
    """LLM-generated analysis of a document's structure and relevant triple patterns.

    Attributes:
        doc_type:                  What kind of document this is
                                   (e.g. "clinical trial report", "10-K filing").
        doc_description:           Short description of the document's content.
        suggested_triple_patterns: Entity-relationship patterns that a knowledge
                                   graph should capture from this kind of document
                                   (e.g. ["company → founded_by → person"]).
    """

    doc_type: str = ""
    doc_description: str = ""
    suggested_triple_patterns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalResult:
    """A single result returned by a similarity search.

    Attributes:
        chunk: The chunk metadata that matched.
        score: Distance/similarity score (lower is better for L2 distance).
    """

    chunk: ChunkMetadata
    score: float
