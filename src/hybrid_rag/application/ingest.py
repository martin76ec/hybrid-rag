"""Ingest document use case.

Orchestrates the ingestion pipeline: read a document, chunk it, embed the
chunks, persist them in the vector store, extract knowledge-graph triples,
refine them (canonical entities, shortened predicates, added/removed triples),
embed canonical entities for semantic matching, and store in the graph store.
"""

from __future__ import annotations

import logging

from ..domain.ports import (
    DocumentReader,
    EmbeddingProvider,
    GraphStore,
    TripleExtractor,
    TripleRefiner,
    VectorStore,
)
from ..domain.services import chunk_text
from ..domain.value_objects import ChunkMetadata, Triple
from .dtos import ExtractionSummary, IngestResult

log = logging.getLogger(__name__)


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
    ) -> None:
        self._reader = reader
        self._embedder = embedder
        self._store = store
        self._triple_extractor = triple_extractor
        self._graph_store = graph_store
        self._triple_refiner = triple_refiner

    def execute(
        self, source: str, *, chunk_size: int = 500, overlap: int = 50
    ) -> IngestResult:
        """Ingest a document from *source*.

        Args:
            source:     File path or URI identifying the document.
            chunk_size: Approximate characters per chunk.
            overlap:    Number of characters to overlap between consecutive chunks.

        Returns:
            An :class:`IngestResult` summarising the operation.
        """
        log.info("Reading document from %s", source)
        doc = self._reader.read(source)

        log.info(
            "Chunking document (%d chars) with chunk_size=%d", len(doc.text), chunk_size
        )
        chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
        log.info("Created %d chunks", len(chunks))

        texts = [c.text for c in chunks]
        log.info("Embedding %d chunks", len(texts))
        vectors = self._embedder.embed_batch(texts)

        metadata = [
            ChunkMetadata(source=c.source, chunk_index=c.chunk_index, text=c.text)
            for c in chunks
        ]

        log.info("Storing %d vectors", len(vectors))
        self._store.add(vectors, metadata)

        num_triples = 0
        summary: ExtractionSummary | None = None

        if self._triple_extractor and self._graph_store:
            log.info("Extracting knowledge-graph triples from %d chunks", len(chunks))
            all_raw: list[Triple] = []
            for c in chunks:
                raw_triples = self._triple_extractor.extract(c.text, source=c.source)
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
            log.info("Extracted %d raw triples", len(all_raw))

            raw_tuples = [(t.subject, t.predicate, t.obj) for t in all_raw]

            if self._triple_refiner and all_raw:
                log.info("Refining %d triples via LLM", len(all_raw))
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
                    refined.append(Triple(subject=s, predicate=p, obj=o, source=source))

                summary = ExtractionSummary(
                    raw_triples=raw_tuples,
                    canonical_mapping=canonical_map,
                    shortened_predicates=result["shortened_predicates"],
                    removed_triples=result["removed_triples"],
                    added_triples=result["added_triples"],
                    refined_triples=[(t.subject, t.predicate, t.obj) for t in refined],
                )
                log.info(
                    "Refined %d raw → %d final triples (removed %d, added %d, canonical %d, shortened %d)",
                    len(all_raw),
                    len(refined),
                    len(result["removed_triples"]),
                    len(result["added_triples"]),
                    len(canonical_map),
                    len(result["shortened_predicates"]),
                )
            else:
                refined = all_raw
                summary = ExtractionSummary(
                    raw_triples=raw_tuples,
                    refined_triples=raw_tuples,
                )

            if refined:
                self._graph_store.add_triples(refined)
                log.info(
                    "Embedding %d canonical entities",
                    len({t.subject for t in refined} | {t.obj for t in refined}),
                )
                entity_names = list(
                    {t.subject for t in refined} | {t.obj for t in refined}
                )
                if entity_names:
                    entity_vectors = self._embedder.embed_batch(entity_names)
                    self._graph_store.set_entity_vectors(
                        dict(zip(entity_names, [v.values for v in entity_vectors]))
                    )
            num_triples = len(refined)

        log.info("Ingestion of %s complete", source)
        return IngestResult(
            source=source,
            num_chunks=len(chunks),
            num_triples=num_triples,
            extraction_summary=summary,
        )
