"""Query knowledge base use case.

Orchestrates the query pipeline: embed the question, retrieve relevant chunks
from the vector store (and optionally the graph store), fuse results with
Reciprocal Rank Fusion, build a prompt with context, and call the LLM to
generate an answer.
"""

from __future__ import annotations

import logging
import time
from typing import Generator

from ..domain.ports import EmbeddingProvider, GraphStore, LanguageModel, VectorStore
from ..domain.services import reciprocal_rank_fusion
from ..domain.value_objects import ChunkMetadata
from .dtos import FusedChunk, QueryDetailedResult, QueryResult, RetrievalStep

log = logging.getLogger(__name__)

# Pipeline step labels (order matters).
QUERY_STEPS = [
    "Embed Question",
    "Vector Search",
    "Graph Search",
    "RRF Fusion",
    "Generate Answer",
]


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
        """Answer *question* using the stored knowledge base (non-streaming).

        Args:
            question: The natural-language question.
            top_k:    Number of retrieved chunks to use as context.

        Returns:
            A :class:`QueryResult` containing the answer and source references.
        """
        log.info("Embedding query")
        t0 = time.perf_counter()
        query_vec = self._embedder.embed(question)
        log.info("Query embedded (%.2fs)", time.perf_counter() - t0)

        log.info("Retrieving top %d chunks from vector store", top_k)
        t0 = time.perf_counter()
        vector_results = self._store.search(query_vec, top_k)
        log.info(
            "Vector search returned %d results (%.2fs)",
            len(vector_results),
            time.perf_counter() - t0,
        )

        ranked_lists: list = [vector_results]

        if self._graph_store:
            log.info("Querying knowledge graph")
            t0 = time.perf_counter()
            graph_results = self._graph_store.query(question, top_k=top_k)
            log.info(
                "Graph query returned %d results (%.2fs)",
                len(graph_results),
                time.perf_counter() - t0,
            )
            if graph_results:
                ranked_lists.append(graph_results)

        if len(ranked_lists) > 1:
            log.info("Fusing %d ranked lists via RRF", len(ranked_lists))
            fused = reciprocal_rank_fusion(ranked_lists)
            results = fused[:top_k]
        else:
            results = vector_results

        prompt = self._build_prompt([r.chunk for r in results], question)
        log.info("Generating answer (prompt_len=%d)", len(prompt))
        t0 = time.perf_counter()
        answer = self._llm.generate(prompt)
        log.info("Answer generated (%.2fs)", time.perf_counter() - t0)

        sources = list(dict.fromkeys(c.source for c in [r.chunk for r in results]))
        return QueryResult(answer=answer, sources=sources)

    def execute_verbose_stepped(
        self,
        question: str,
        *,
        top_k: int = 5,
    ) -> Generator[tuple[int, str], None, QueryDetailedResult]:
        """Answer *question* with full retrieval provenance, yielding step updates.

        Callers **must** iterate this generator to completion to run the full
        pipeline. The final :class:`QueryDetailedResult` is returned as the
        generator return value (accessible via ``return_value`` after iteration
        completes).

        Yields:
            ``(step_index, step_label)`` after each pipeline step completes.
        """
        query_vec = self._embedder.embed(question)
        yield (0, QUERY_STEPS[0])

        vector_results = self._store.search(query_vec, top_k)

        vector_steps = [
            RetrievalStep(
                source=r.chunk.source,
                chunk_index=r.chunk.chunk_index,
                text=r.chunk.text,
                rank=i + 1,
                score=r.score,
            )
            for i, r in enumerate(vector_results)
        ]
        yield (1, QUERY_STEPS[1])

        ranked_lists: list = [vector_results]
        graph_steps: list[RetrievalStep] = []
        graph_entities: list[str] = []

        if self._graph_store:
            graph_results = self._graph_store.query(question, top_k=top_k)
            log.info("Graph search returned %d results", len(graph_results))
            graph_entities = self._graph_store.extract_entity_mentions(question)
            graph_steps = [
                RetrievalStep(
                    source=r.chunk.source,
                    chunk_index=r.chunk.chunk_index,
                    text=r.chunk.text,
                    rank=i + 1,
                    score=r.score,
                )
                for i, r in enumerate(graph_results)
            ]
            if graph_results:
                ranked_lists.append(graph_results)
        yield (2, QUERY_STEPS[2])

        if len(ranked_lists) > 1:
            fused = reciprocal_rank_fusion(ranked_lists)
        else:
            fused = reciprocal_rank_fusion([vector_results])
        yield (3, QUERY_STEPS[3])

        fused_results = fused[:top_k]

        vector_rank_map = {(s.source, s.chunk_index): s.rank for s in vector_steps}
        graph_rank_map = {(s.source, s.chunk_index): s.rank for s in graph_steps}

        fused_chunks: list[FusedChunk] = []
        for i, r in enumerate(fused_results):
            key = (r.chunk.source, r.chunk.chunk_index)
            v_rank = vector_rank_map.get(key, 0)
            g_rank = graph_rank_map.get(key, 0)
            paths = []
            if v_rank:
                paths.append("vector")
            if g_rank:
                paths.append("graph")
            fused_chunks.append(
                FusedChunk(
                    source=r.chunk.source,
                    chunk_index=r.chunk.chunk_index,
                    text=r.chunk.text,
                    rrf_score=round(r.score, 4),
                    vector_rank=v_rank,
                    graph_rank=g_rank,
                    path=" + ".join(paths) if paths else "vector",
                )
            )

        prompt = self._build_prompt([r.chunk for r in fused_results[:top_k]], question)
        answer = self._llm.generate(prompt)
        yield (4, QUERY_STEPS[4])

        return QueryDetailedResult(
            answer=answer,
            vector_results=vector_steps,
            graph_results=graph_steps,
            fused_results=fused_chunks,
            graph_entities=graph_entities,
        )

    def execute_verbose(
        self,
        question: str,
        *,
        top_k: int = 5,
    ) -> QueryDetailedResult:
        """Answer *question* with full retrieval provenance (non-streaming).

        Runs the full pipeline synchronously without yielding intermediate
        step updates. Equivalent to consuming :meth:`execute_verbose_stepped`
        without inspecting the yields.

        Args:
            question:  The natural-language question.
            top_k:     Number of retrieved chunks to use as context.

        Returns:
            A :class:`QueryDetailedResult` containing the LLM answer plus
            the intermediate retrieval results from each path and the fused output.
        """
        gen = self.execute_verbose_stepped(question, top_k=top_k)
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as exc:
            result = exc.value
        return result

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
