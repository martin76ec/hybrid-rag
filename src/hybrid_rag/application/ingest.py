"""Ingest document use case.

Orchestrates the ingestion pipeline: read a document, chunk it, embed the
chunks, and persist them in the vector store.
"""

from __future__ import annotations

import logging

from ..domain.entities import Document
from ..domain.ports import DocumentReader, EmbeddingProvider, VectorStore
from ..domain.services import chunk_text
from ..domain.value_objects import ChunkMetadata
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
    ) -> None:
        self._reader = reader
        self._embedder = embedder
        self._store = store

    def execute(self, source: str, *, chunk_size: int = 500, overlap: int = 50) -> IngestResult:
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

        log.info("Chunking document (%d chars) with chunk_size=%d", len(doc.text), chunk_size)
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

        log.info("Ingestion of %s complete", source)
        return IngestResult(source=source, num_chunks=len(chunks))