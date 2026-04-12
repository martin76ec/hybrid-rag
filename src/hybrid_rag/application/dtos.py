"""Data transfer objects for the application layer.

DTOs are simple data containers used to pass data between the application and
presentation layers without leaking domain internals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestResult:
    """Result of a document ingestion use case.

    Attributes:
        source:      The source identifier of the ingested document.
        num_chunks:  Number of chunks produced.
        num_triples: Number of knowledge-graph triples extracted.
    """

    source: str
    num_chunks: int
    num_triples: int = 0


@dataclass(frozen=True)
class QueryResult:
    """Result of a knowledge-base query use case.

    Attributes:
        answer:  The LLM-generated answer.
        sources: Source identifiers of the chunks used as context.
    """

    answer: str
    sources: list[str]
