"""FAISS vector store — implements :class:`VectorStore`.

Persists vectors in a FAISS ``IndexFlatL2`` on disk and stores associated
metadata as a JSONL file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import faiss
import numpy as np

from ...domain.ports import VectorStore
from ...domain.value_objects import ChunkMetadata, EmbeddingVector, RetrievalResult

log = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """Concrete :class:`VectorStore` backed by a FAISS ``IndexFlatL2`` on disk."""

    def __init__(self, index_path: str) -> None:
        self._index_path = Path(index_path)

    # -- Public interface -----------------------------------------------------

    def add(
        self, vectors: list[EmbeddingVector], metadata: list[ChunkMetadata]
    ) -> None:
        """Add vectors and metadata to the FAISS index and persist to disk."""
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have the same length")
        if not vectors:
            return

        log.info("Adding %d vectors to FAISS index", len(vectors))
        matrix = np.array([v.values for v in vectors], dtype=np.float32)
        dim = matrix.shape[1]
        index = self._load_or_create_index(dim)
        index.add(matrix)

        self._index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self._index_path / "index"))

        meta_path = self._index_path / "metadata.jsonl"
        with meta_path.open("a", encoding="utf-8") as f:
            for item in metadata:
                f.write(
                    json.dumps(
                        {
                            "source": item.source,
                            "chunk_index": item.chunk_index,
                            "text": item.text,
                        }
                    )
                    + "\n"
                )

    def search(self, query: EmbeddingVector, top_k: int) -> list[RetrievalResult]:
        """Search the FAISS index for the *top_k* nearest neighbours."""
        log.info("Searching FAISS index top_k=%d", top_k)
        index, metadata = self._load_index_and_metadata()
        query_vec = np.array([query.values], dtype=np.float32)
        distances, indices = index.search(query_vec, top_k)

        results: list[RetrievalResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(metadata):
                results.append(RetrievalResult(chunk=metadata[idx], score=float(score)))
        log.info("Search returned %d results", len(results))
        return results

    # -- Private helpers ------------------------------------------------------

    def _load_or_create_index(self, dim: int) -> faiss.IndexFlatL2:
        """Load an existing index or create a new one."""
        index_file = self._index_path / "index"
        if index_file.is_file():
            index = faiss.read_index(str(index_file))
            if index.d != dim:
                raise ValueError(
                    f"Existing index dimension {index.d} does not match embedding dimension {dim}"
                )
            return index
        return faiss.IndexFlatL2(dim)

    def _load_index_and_metadata(self) -> tuple[faiss.IndexFlatL2, list[ChunkMetadata]]:
        """Load the FAISS index and its associated metadata from disk."""
        index = faiss.read_index(str(self._index_path / "index"))
        metadata: list[ChunkMetadata] = []
        meta_path = self._index_path / "metadata.jsonl"
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                metadata.append(
                    ChunkMetadata(
                        source=obj["source"],
                        chunk_index=obj["chunk_index"],
                        text=obj["text"],
                    )
                )
        return index, metadata
