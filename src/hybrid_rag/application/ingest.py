"""Ingest document use case.

Orchestrates the ingestion pipeline: read a document, chunk it, embed the
chunks, persist them in the vector store, analyse the document structure,
extract knowledge-graph triples (guided by the analysis), refine them
(canonical entities, shortened predicates, added/removed triples), embed
canonical entities for semantic matching, and store in the graph store.
"""

from __future__ import annotations

import logging
import time
from typing import Generator

from ..domain.ports import (
    DocumentAnalyzer,
    DocumentReader,
    EmbeddingProvider,
    GraphStore,
    TripleExtractor,
    TripleRefiner,
    VectorStore,
)
from ..domain.services import chunk_text
from ..domain.value_objects import ChunkMetadata, DocumentAnalysis, Triple
from .dtos import ExtractionSummary, IngestResult

log = logging.getLogger(__name__)

# Pipeline step labels (order matters).
INGEST_STEPS = [
    "Read & Chunk",
    "Embed Chunks",
    "Analyze Document",
    "Extract Triples",
    "Refine Triples",
    "Embed Entities",
    "Build Graph",
]


class IngestDocumentUseCase:
    """Application service that ingests a single document into the knowledge base.

    Dependencies are injected through the constructor so the use case is
    decoupled from any specific infrastructure implementation.
    """

    def __init__(
        self,
        reader: DocumentReader,
        embedder: EmbeddingProvider,
        store: VectorStore,
        triple_extractor: TripleExtractor | None = None,
        graph_store: GraphStore | None = None,
        triple_refiner: TripleRefiner | None = None,
        document_analyzer: DocumentAnalyzer | None = None,
    ) -> None:
        self._reader = reader
        self._embedder = embedder
        self._store = store
        self._triple_extractor = triple_extractor
        self._graph_store = graph_store
        self._triple_refiner = triple_refiner
        self._document_analyzer = document_analyzer

    def execute_stepped(
        self,
        source: str,
        *,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> Generator[tuple[int, str], None, IngestResult]:
        """Ingest a document, yielding ``(step_index, step_label)`` after each step.

        Callers **must** iterate this generator to completion to run the full
        pipeline. The final :class:`IngestResult` is returned as the generator
        return value (accessible via ``return_value`` after iteration completes).

        Yields:
            ``(step_index, step_label)`` after each pipeline step completes.
        """
        log.info("Reading document from %s", source)
        t0 = time.perf_counter()
        doc = self._reader.read(source)

        log.info(
            "Chunking document (%d chars) with chunk_size=%d", len(doc.text), chunk_size
        )
        chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
        log.info("Created %d chunks (%.2fs)", len(chunks), time.perf_counter() - t0)
        yield (0, INGEST_STEPS[0])

        texts = [c.text for c in chunks]
        log.info("Embedding %d chunks", len(texts))
        t0 = time.perf_counter()
        vectors = self._embedder.embed_batch(texts)

        metadata = [
            ChunkMetadata(source=c.source, chunk_index=c.chunk_index, text=c.text)
            for c in chunks
        ]

        log.info("Storing %d vectors", len(vectors))
        self._store.add(vectors, metadata)
        log.info("Vectors stored (%.2fs)", time.perf_counter() - t0)
        yield (1, INGEST_STEPS[1])

        num_triples = 0
        summary: ExtractionSummary | None = None
        doc_analysis: DocumentAnalysis | None = None

        if self._triple_extractor and self._graph_store:
            if self._document_analyzer:
                log.info("Analyzing document structure for extraction guidance")
                t0 = time.perf_counter()
                doc_analysis = self._document_analyzer.analyze(doc.text)
                log.info(
                    "Document analysis: type=%s (%.2fs)",
                    doc_analysis.doc_type,
                    time.perf_counter() - t0,
                )
                self._triple_extractor.set_guidance(doc_analysis)
            yield (2, INGEST_STEPS[2])

            log.info("Extracting knowledge-graph triples from %d chunks", len(chunks))
            t0 = time.perf_counter()
            all_raw: list[Triple] = []
            for c in chunks:
                raw_triples = self._triple_extractor.extract(c.text, source=c.source)
                for t in raw_triples:
                    log.debug(
                        "  raw triple: (%s, %s, %s)", t.subject, t.predicate, t.obj
                    )
                enriched = [
                    Triple(
                        subject=t.subject,
                        predicate=t.predicate,
                        obj=t.obj,
                        source=c.source,
                        chunk_index=c.chunk_index,
                        chunk_text=c.text,
                    )
                    for t in raw_triples
                ]
                all_raw.extend(enriched)
            log.info(
                "Extracted %d raw triples (%.2fs)",
                len(all_raw),
                time.perf_counter() - t0,
            )
            yield (3, INGEST_STEPS[3])

            raw_tuples = [(t.subject, t.predicate, t.obj) for t in all_raw]

            if self._triple_refiner and all_raw:
                log.info("Refining %d triples via LLM", len(all_raw))
                t0 = time.perf_counter()
                result = self._triple_refiner.refine(all_raw)
                canonical_map = result["canonical_mapping"]
                shortened = dict(result["shortened_predicates"])
                removed_set = set(result["removed_triples"])

                refined: list[Triple] = []
                for t in all_raw:
                    triple_key = (t.subject, t.predicate, t.obj)
                    if triple_key in removed_set:
                        continue
                    s = canonical_map.get(t.subject, t.subject)
                    o = canonical_map.get(t.obj, t.obj)
                    p = shortened.get(t.predicate, t.predicate)
                    if s and p and o and s.lower() != "null" and o.lower() != "null":
                        refined.append(
                            Triple(
                                subject=s,
                                predicate=p,
                                obj=o,
                                source=t.source,
                                chunk_index=t.chunk_index,
                                chunk_text=t.chunk_text,
                            )
                        )

                for s, p, o in result["added_triples"]:
                    log.debug("  added triple: (%s, %s, %s)", s, p, o)
                    refined.append(Triple(subject=s, predicate=p, obj=o, source=source))

                summary = ExtractionSummary(
                    doc_type=doc_analysis.doc_type if doc_analysis else "",
                    doc_description=doc_analysis.doc_description
                    if doc_analysis
                    else "",
                    suggested_triple_patterns=doc_analysis.suggested_triple_patterns
                    if doc_analysis
                    else [],
                    raw_triples=raw_tuples,
                    canonical_mapping=canonical_map,
                    shortened_predicates=result["shortened_predicates"],
                    removed_triples=result["removed_triples"],
                    added_triples=result["added_triples"],
                    refined_triples=[(t.subject, t.predicate, t.obj) for t in refined],
                )
                log.info(
                    "Refined %d raw → %d final triples (removed %d, added %d, canonical %d, shortened %d) (%.2fs)",
                    len(all_raw),
                    len(refined),
                    len(result["removed_triples"]),
                    len(result["added_triples"]),
                    len(canonical_map),
                    len(result["shortened_predicates"]),
                    time.perf_counter() - t0,
                )
            else:
                refined = all_raw
                summary = ExtractionSummary(
                    doc_type=doc_analysis.doc_type if doc_analysis else "",
                    doc_description=doc_analysis.doc_description
                    if doc_analysis
                    else "",
                    suggested_triple_patterns=doc_analysis.suggested_triple_patterns
                    if doc_analysis
                    else [],
                    raw_triples=raw_tuples,
                    refined_triples=raw_tuples,
                )
            yield (4, INGEST_STEPS[4])

            if refined:
                self._graph_store.add_triples(refined)
                log.info(
                    "Embedding %d canonical entities",
                    len({t.subject for t in refined} | {t.obj for t in refined}),
                )
                t0 = time.perf_counter()
                entity_names = list(
                    {t.subject for t in refined} | {t.obj for t in refined}
                )
                if entity_names:
                    entity_vectors = self._embedder.embed_batch(entity_names)
                    self._graph_store.set_entity_vectors(
                        dict(zip(entity_names, [v.values for v in entity_vectors]))
                    )
                    log.info("Entity vectors stored (%.2fs)", time.perf_counter() - t0)
            num_triples = len(refined)
            yield (5, INGEST_STEPS[5])

            log.info("Building knowledge graph")
            yield (6, INGEST_STEPS[6])

        log.info("Ingestion of %s complete", source)
        return IngestResult(
            source=source,
            num_chunks=len(chunks),
            num_triples=num_triples,
            extraction_summary=summary,
        )

    def execute(
        self,
        source: str,
        *,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> IngestResult:
        """Ingest a document from *source* (non-streaming convenience wrapper).

        Runs the full pipeline synchronously without yielding intermediate
        step updates. Equivalent to consuming :meth:`execute_stepped` without
        inspecting the yields.

        Args:
            source:     File path or URI identifying the document.
            chunk_size: Approximate characters per chunk.
            overlap:    Number of characters to overlap between consecutive chunks.

        Returns:
            An :class:`IngestResult` summarising the operation.
        """
        gen = self.execute_stepped(source, chunk_size=chunk_size, overlap=overlap)
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as exc:
            result = exc.value
        return result
