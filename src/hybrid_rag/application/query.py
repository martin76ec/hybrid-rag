"""Query knowledge base use case.

Orchestrates the query pipeline: embed the question, retrieve relevant chunks
from the vector store (and optionally the graph store), fuse results with
Reciprocal Rank Fusion, build a prompt with context, and call the LLM to
generate an answer.
"""

from __future__ import annotations

import logging

from ..domain.ports import EmbeddingProvider, GraphStore, LanguageModel, VectorStore
from ..domain.services import reciprocal_rank_fusion
from ..domain.value_objects import ChunkMetadata
from .dtos import QueryResult

log = logging.getLogger(__name__)


class QueryKnowledgeBaseUseCase:
    """Application service that answers a question using the stored knowledge base.

    Supports hybrid retrieval: when a :class:`GraphStore` is provided, results
    from vector search and graph neighbour traversal are fused with RRF.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LanguageModel,
        graph_store: GraphStore | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._graph_store = graph_store

    def execute(self, question: str, *, top_k: int = 5) -> QueryResult:
        """Answer *question* using the stored knowledge base.

        Args:
            question: The natural-language question.
            top_k:    Number of retrieved chunks to use as context.

        Returns:
            A :class:`QueryResult` containing the answer and source references.
        """
        log.info("Embedding query")
        query_vec = self._embedder.embed(question)

        log.info("Retrieving top %d chunks from vector store", top_k)
        vector_results = self._store.search(query_vec, top_k)

        ranked_lists: list = [vector_results]

        if self._graph_store:
            log.info("Querying knowledge graph")
            graph_results = self._graph_store.query(question, top_k=top_k)
            if graph_results:
                ranked_lists.append(graph_results)
                log.info("Graph returned %d results", len(graph_results))

        if len(ranked_lists) > 1:
            log.info("Fusing %d ranked lists via RRF", len(ranked_lists))
            fused = reciprocal_rank_fusion(ranked_lists)
            results = fused[:top_k]
        else:
            results = vector_results

        prompt = self._build_prompt([r.chunk for r in results], question)
        log.debug("Prompt built, calling LLM")
        answer = self._llm.generate(prompt)

        sources = list(dict.fromkeys(c.source for c in [r.chunk for r in results]))
        return QueryResult(answer=answer, sources=sources)

    @staticmethod
    def _build_prompt(context_chunks: list[ChunkMetadata], question: str) -> str:
        """Build a prompt string from context chunks and a question."""
        parts = [
            "You are a helpful assistant that answers questions using the provided context.\n"
        ]
        for i, chunk in enumerate(context_chunks, 1):
            parts.append(
                f"--- Context {i} (from {chunk.source} chunk {chunk.chunk_index}) ---\n"
                f"{chunk.text}\n"
            )
        parts.append("--- Question ---\n" + question + "\n")
        parts.append("Answer succinctly using only the above information.")
        return "\n".join(parts)
