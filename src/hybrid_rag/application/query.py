"""Query knowledge base use case.

Orchestrates the query pipeline: embed the question, retrieve relevant chunks,
build a prompt with context, and call the LLM to generate an answer.
"""

from __future__ import annotations

import logging

from ..domain.ports import EmbeddingProvider, LanguageModel, VectorStore
from ..domain.value_objects import ChunkMetadata
from .dtos import QueryResult

log = logging.getLogger(__name__)


class QueryKnowledgeBaseUseCase:
    """Application service that answers a question using the stored knowledge base.

    Dependencies are injected through the constructor.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LanguageModel,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm

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

        log.info("Retrieving top %d chunks", top_k)
        results = self._store.search(query_vec, top_k)

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