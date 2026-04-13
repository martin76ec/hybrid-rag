"""Data transfer objects for the application layer.

DTOs are simple data containers used to pass data between the application and
presentation layers without leaking domain internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractionSummary:
    """Step-by-step summary of the extraction pipeline for display.

    Attributes:
        raw_triples:         All raw triples before refinement.
        canonical_mapping:   Entity names mapped to their canonical form.
                             e.g. {"NovaMind": "NovaMind Technologies"}.
        shortened_predicates: Before→after pairs for predicate shortening.
        removed_triples:     Triples judged trivial or useless by the refiner.
        added_triples:       Triples the refiner suggested as missing.
        refined_triples:     Final triples after all corrections.
    """

    raw_triples: list[tuple[str, str, str]] = field(default_factory=list)
    canonical_mapping: dict[str, str] = field(default_factory=dict)
    shortened_predicates: list[tuple[str, str]] = field(default_factory=list)
    removed_triples: list[tuple[str, str, str]] = field(default_factory=list)
    added_triples: list[tuple[str, str, str]] = field(default_factory=list)
    refined_triples: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class IngestResult:
    source: str
    num_chunks: int
    num_triples: int = 0
    extraction_summary: ExtractionSummary | None = None


@dataclass(frozen=True)
class QueryResult:
    """Result of a knowledge-base query use case.

    Attributes:
        answer:  The LLM-generated answer.
        sources: Source identifiers of the chunks used as context.
    """

    answer: str
    sources: list[str]


@dataclass(frozen=True)
class RetrievalStep:
    """A single chunk's retrieval information for one path.

    Attributes:
        source:      Source file of the chunk.
        chunk_index: Index of the chunk within the source.
        text:        Chunk text (truncated for display).
        rank:        Position in this path's ranked list (1-based).
        score:       Raw retrieval score from this path.
    """

    source: str
    chunk_index: int
    text: str
    rank: int
    score: float


@dataclass(frozen=True)
class FusedChunk:
    """A chunk after reciprocal rank fusion with provenance info.

    Attributes:
        source:       Source file of the chunk.
        chunk_index:  Index of the chunk within the source.
        text:         Chunk text.
        rrf_score:    Final fused RRF score.
        vector_rank:  Rank in vector path (0 = not present).
        graph_rank:   Rank in graph path (0 = not present).
        path:         Which paths contributed: 'vector', 'graph', or 'both'.
    """

    source: str
    chunk_index: int
    text: str
    rrf_score: float
    vector_rank: int = 0
    graph_rank: int = 0
    path: str = ""


@dataclass(frozen=True)
class QueryDetailedResult:
    """Verbose query result exposing intermediate retrieval states.

    Attributes:
        answer:         The LLM-generated answer.
        vector_results: Chunks retrieved by the vector path.
        graph_results:  Chunks retrieved by the graph path.
        fused_results:  Chunks after RRF fusion.
        graph_entities: Entity nodes matched from the query.
    """

    answer: str
    vector_results: list[RetrievalStep]
    graph_results: list[RetrievalStep]
    fused_results: list[FusedChunk]
    graph_entities: list[str]
