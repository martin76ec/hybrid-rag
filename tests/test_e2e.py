"""End-to-end tests with retrieval, generation, and knowledge-graph quality metrics.

These tests exercise the full hybrid pipeline against real Ollama, FAISS,
and NetworkX, measuring:

- **Retrieval Precision @ K**: Do the top-k retrieved chunks contain the
  right source document for the question?
- **Answer Correctness (keyword overlap)**: Does the generated answer
  contain key entities/facts expected from the source?
- **Answer Groundedness**: Does the answer stay within the provided
  context (no hallucinated numbers)?
- **Graph Extraction**: Does the triple extractor produce knowledge-graph
  triples from ingested documents?
- **Hybrid Retrieval**: Does the graph store contribute to retrieval
  (non-zero node/edge counts)?

Run with:  pytest tests/test_e2e.py -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from hybrid_rag.application.ingest import IngestDocumentUseCase
from hybrid_rag.application.query import QueryKnowledgeBaseUseCase
from hybrid_rag.infrastructure.config import Config
from hybrid_rag.infrastructure.faiss import FAISSVectorStore
from hybrid_rag.infrastructure.networkx import NetworkXGraphStore
from hybrid_rag.infrastructure.ollama import (
    OllamaEmbeddingProvider,
    OllamaLanguageModel,
    OllamaTripleExtractor,
)
from hybrid_rag.infrastructure.pypdf import PyPDFDocumentReader
from hybrid_rag.domain.value_objects import ChunkMetadata

from fixtures.pdf_factory import generate_all_pdfs, FIXTURES_DIR


def _keyword_overlap(answer: str, keywords: list[str]) -> float:
    hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return hits / len(keywords) if keywords else 0.0


def _chunks_contain_keywords(chunks: list[ChunkMetadata], keywords: list[str]) -> float:
    hits = sum(
        1 for kw in keywords if any(kw.lower() in c.text.lower() for c in chunks)
    )
    return hits / len(keywords) if keywords else 0.0


@pytest.fixture(scope="module")
def e2e_env():
    pdf_paths = generate_all_pdfs()
    cfg = Config()
    index_dir = tempfile.mkdtemp(prefix="hybrid_rag_e2e_")
    graph_dir = tempfile.mkdtemp(prefix="hybrid_rag_graph_")

    embedder = OllamaEmbeddingProvider(cfg.ollama_host, cfg.embedding_model)
    store = FAISSVectorStore(index_dir)
    reader = PyPDFDocumentReader()
    llm = OllamaLanguageModel(cfg.ollama_host, cfg.llm_model)
    triple_extractor = OllamaTripleExtractor(cfg.ollama_host, cfg.llm_model)
    graph_store = NetworkXGraphStore(graph_dir)

    ingest_uc = IngestDocumentUseCase(
        reader=reader,
        embedder=embedder,
        store=store,
        triple_extractor=triple_extractor,
        graph_store=graph_store,
    )
    query_uc = QueryKnowledgeBaseUseCase(
        embedder=embedder,
        store=store,
        llm=llm,
        graph_store=graph_store,
    )

    for pdf in pdf_paths:
        ingest_uc.execute(str(pdf), chunk_size=400, overlap=60)

    yield {
        "query_uc": query_uc,
        "pdf_paths": pdf_paths,
        "index_dir": index_dir,
        "graph_dir": graph_dir,
        "graph_store": graph_store,
    }

    shutil.rmtree(index_dir, ignore_errors=True)
    shutil.rmtree(graph_dir, ignore_errors=True)
    for p in pdf_paths:
        p.unlink(missing_ok=True)


class TestRetrievalQuality:
    """Measure whether the right chunks are retrieved for targeted questions."""

    QUESTIONS = [
        {
            "question": "Who founded NovaMind Technologies?",
            "expected_source_substring": "tech_company",
            "expected_keywords": ["Elena Vasquez", "Raj Patel", "Austin"],
        },
        {
            "question": "What is the power consumption of the Cortex-7 processor?",
            "expected_source_substring": "tech_company",
            "expected_keywords": ["15 watts", "Cortex-7", "120 tera-operations"],
        },
        {
            "question": "What was the primary endpoint of the VXC-204 trial?",
            "expected_source_substring": "clinical_trial",
            "expected_keywords": ["FVC", "percentage points", "52"],
        },
        {
            "question": "Who is the principal investigator of the VXC-204 study?",
            "expected_source_substring": "clinical_trial",
            "expected_keywords": ["Amara Okafor", "Michigan"],
        },
        {
            "question": "What is NovaMind's target IPO valuation?",
            "expected_source_substring": "tech_company",
            "expected_keywords": ["5.8 billion", "Q3 2025"],
        },
    ]

    @pytest.mark.parametrize("case", QUESTIONS, ids=lambda c: c["question"][:40])
    def test_retrieval_precision(self, e2e_env, case):
        uc = e2e_env["query_uc"]
        result = uc.execute(case["question"], top_k=5)
        sources = result.sources
        hit = any(case["expected_source_substring"] in s for s in sources)
        assert hit, (
            f"Retrieval precision FAILED for '{case['question']}': "
            f"expected source containing '{case['expected_source_substring']}', "
            f"got sources={sources}"
        )

    @pytest.mark.parametrize("case", QUESTIONS, ids=lambda c: c["question"][:40])
    def test_answer_keyword_overlap(self, e2e_env, case):
        uc = e2e_env["query_uc"]
        result = uc.execute(case["question"], top_k=5)
        overlap = _keyword_overlap(result.answer, case["expected_keywords"])
        assert overlap >= 0.3, (
            f"Answer keyword overlap {overlap:.0%} < 30% for '{case['question']}': "
            f"expected keywords {case['expected_keywords']}, "
            f"answer='{result.answer[:200]}'"
        )


class TestAnswerGroundedness:
    """Verify the LLM doesn't fabricate numbers outside the source context."""

    GROUNDEDNESS_QUESTIONS = [
        {
            "question": "How much revenue did NovaMind report in fiscal year 2024?",
            "ground_truth_number": "$340 million",
            "wrong_numbers": ["$500 million", "$200 million", "$1 billion"],
        },
        {
            "question": "How many participants were enrolled in the VXC-204 trial?",
            "ground_truth_number": "648",
            "wrong_numbers": ["1000", "500", "2000"],
        },
    ]

    @pytest.mark.parametrize(
        "case", GROUNDEDNESS_QUESTIONS, ids=lambda c: c["question"][:40]
    )
    def test_answer_uses_correct_numbers(self, e2e_env, case):
        uc = e2e_env["query_uc"]
        result = uc.execute(case["question"], top_k=5)
        answer = result.answer.lower()
        assert case["ground_truth_number"].lower() in answer, (
            f"Groundedness FAILED: expected '{case['ground_truth_number']}' in answer, "
            f"got: '{result.answer[:300]}'"
        )
        for wrong in case["wrong_numbers"]:
            assert wrong.lower() not in answer, (
                f"Hallucination detected: wrong number '{wrong}' appeared in answer "
                f"for question '{case['question']}'"
            )


class TestKnowledgeGraph:
    """Verify knowledge graph extraction and structure."""

    def test_graph_has_nodes(self, e2e_env):
        graph_store: NetworkXGraphStore = e2e_env["graph_store"]
        assert graph_store.node_count() > 0, "Graph should contain at least some nodes"

    def test_graph_has_edges(self, e2e_env):
        graph_store: NetworkXGraphStore = e2e_env["graph_store"]
        assert graph_store.edge_count() > 0, (
            "Graph should contain at least some edges (triples)"
        )

    def test_graph_persistence(self, e2e_env):
        graph_dir = e2e_env["graph_dir"]
        assert (Path(graph_dir) / "graph.json").is_file(), (
            "Graph JSON should be persisted"
        )

    def test_triples_contain_known_entities(self, e2e_env):
        graph_store: NetworkXGraphStore = e2e_env["graph_store"]
        triples = graph_store.all_triples()
        triple_text = " ".join(f"{t.subject} {t.predicate} {t.obj}" for t in triples)
        known_entities = ["NovaMind", "Cortex-7", "VXC-204", "Veratralimab"]
        hits = sum(1 for e in known_entities if e.lower() in triple_text.lower())
        assert hits >= 1, (
            f"Expected at least 1 known entity in graph triples, found {hits}/4. "
            f"Entities checked: {known_entities}"
        )

    def test_graph_enriches_retrieval(self, e2e_env):
        graph_store: NetworkXGraphStore = e2e_env["graph_store"]
        results = graph_store.query("NovaMind Cortex-7 processor", top_k=5)
        assert len(results) > 0, "Graph query should return results for known entities"


class TestIngestionMetrics:
    """Basic pipeline health — ingestion produces reasonable chunks."""

    def test_all_pdfs_ingested(self, e2e_env):
        pdf_paths: list[Path] = e2e_env["pdf_paths"]
        assert len(pdf_paths) == 2

    def test_index_directory_exists(self, e2e_env):
        index_dir = e2e_env["index_dir"]
        assert Path(index_dir).is_dir()
        assert (Path(index_dir) / "index").is_file()
        assert (Path(index_dir) / "metadata.jsonl").is_file()

    def test_metadata_has_chunks(self, e2e_env):
        meta_path = Path(e2e_env["index_dir"]) / "metadata.jsonl"
        lines = meta_path.read_text().strip().split("\n")
        assert len(lines) >= 4, f"Expected >= 4 chunks total, got {len(lines)}"
