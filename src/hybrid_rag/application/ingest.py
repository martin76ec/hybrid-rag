"""Ingest document use case.

Orchestrates the ingestion pipeline: read a document, chunk it, embed the
chunks, persist them in the vector store, extract knowledge-graph triples,
and store them in the graph store.
"""

from __future__ import annotations

import logging

from ..domain.ports import (
    DocumentReader,
    EmbeddingProvider,
    GraphStore,
    TripleExtractor,
    VectorStore,
)
from ..domain.services import chunk_text
from ..domain.value_objects import ChunkMetadata, Triple
from .dtos import IngestResult

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
    ) -> None:
        self._reader = reader
        self._embedder = embedder
        self._store = store
        self._triple_extractor = triple_extractor
        self._graph_store = graph_store

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
        if self._triple_extractor and self._graph_store:
            log.info("Extracting knowledge-graph triples from %d chunks", len(chunks))
            all_triples: list[Triple] = []
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
                all_triples.extend(enriched)
            log.info("Extracted %d triples total", len(all_triples))
            if all_triples:
                self._graph_store.add_triples(all_triples)
            num_triples = len(all_triples)

        log.info("Ingestion of %s complete", source)
        return IngestResult(
            source=source, num_chunks=len(chunks), num_triples=num_triples
        )
