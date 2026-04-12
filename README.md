<div align="center">

# Hybrid-RAG

**Retrieval-Augmented Generation — reimagined with hybrid search.**

Vector similarity + knowledge graph retrieval, layered under clean architecture.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: None](https://img.shields.io/badge/license-unlicensed-lightgrey)]()
[![Code Style: DDD](https://img.shields.io/badge/architecture-DDD%20%2F%20Ports%20%26%20Adapters-blueviolet)]()

</div>

---

## Why Hybrid-RAG?

Most RAG pipelines stop at vector search. Hybrid-RAG is built to combine **dense vector retrieval** with **knowledge graph traversal** — two complementary paradigms that catch different kinds of relevance:

| Retrieval Type | Good At | Misses |
|---|---|---|
| Vector Search (FAISS) | Semantic similarity, fuzzy concepts | Exact relationships, multi-hop reasoning |
| Knowledge Graph (planned) | Structured relationships, multi-hop queries | Vague or paraphrased concepts |

Hybrid-RAG runs both and fuses the results — so you get the best of both worlds.

> **Status**: Vector retrieval is fully functional. Knowledge graph retrieval is on the roadmap (see [Roadmap](#roadmap)).

---

## Architecture

Hybrid-RAG follows **Domain-Driven Design** with a strict **Ports & Adapters** (Hexagonal) architecture. Every infrastructure dependency is hidden behind an abstract port, making the core domain fully testable and framework-agnostic.

```
┌─────────────────────────────────────────────┐
│              Presentation (CLI)              │
│                   Typer                      │
├─────────────────────────────────────────────┤
│            Application (Use Cases)           │
│        IngestDocument · QueryKnowledge       │
├─────────────────────────────────────────────┤
│               Domain (Core)                  │
│   Entities · Value Objects · Ports · Services │
├──────────┬──────────┬──────────┬────────────┤
│  Ollama  │  FAISS   │  PyPDF   │  NetworkX  │
│  Embed   │  Index   │  Reader  │   Graph    │
│  Ollama  │  Store   │          │   Store    │
│   LLM    │          │          │  (planned) │
└──────────┴──────────┴──────────┴────────────┘
```

**Key design choices:**

- **No LangChain. No LlamaIndex.** Raw HTTP to Ollama, direct FAISS — zero AI-framework bloat.
- **Fully swappable adapters.** Swap FAISS for Qdrant, Ollama for OpenAI, PyPDF for Unstructured — without touching business logic.
- **Testable by design.** Smoke tests use fake implementations that pass without any external service running.

---

## Features

- **PDF Ingestion** — Extract text from any PDF and chunk it into overlapping segments
- **Overlapping Chunks** — Configurable chunk size and overlap to preserve context across boundaries
- **Vector Embedding** — Dense embeddings via Ollama (default: `nomic-embed-text`)
- **FAISS Index** — Persistent L2 similarity index with JSONL metadata sidecar
- **LLM-Powered Q&A** — Generate grounded answers from retrieved context via Ollama (default: `phi3`)
- **Source Attribution** — Every answer includes the source chunks used to generate it
- **CLI Interface** — Beautiful terminal output with `rich` formatting
- **Full Config** — Environment-based configuration, zero hard-coded paths

---

## Quick Start

### Prerequisites

| Requirement | Install |
|---|---|
| Python ≥ 3.10 | [python.org](https://www.python.org/) |
| [Ollama](https://ollama.ai) | `curl -fsSL https://ollama.ai/install.sh \| sh` |
| [uv](https://github.com/astral-sh/uv) (recommended) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

### 1. Pull the models

```bash
ollama pull nomic-embed-text
ollama pull phi3
```

### 2. Install Hybrid-RAG

```bash
git clone https://github.com/your-org/hybrid-rag.git && cd hybrid-rag
uv sync           # or: pip install -e .
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — defaults work out of the box with local Ollama
```

### 4. Ingest & Query

```bash
# Ingest PDFs into the vector store
hybrid-rag ingest ./papers/attention-is-all-you-need.pdf

# Ask questions
hybrid-rag query "What is the main contribution of the Transformer architecture?"
```

---

## Usage

### Ingest

```bash
# Single document
hybrid-rag ingest ./document.pdf

# Multiple documents
hybrid-rag ingest ./doc1.pdf ./doc2.pdf ./doc3.pdf

# Custom chunk size
hybrid-rag ingest ./document.pdf --chunk-size 1000
```

### Query

```bash
# Basic query (returns top 5 chunks as context)
hybrid-rag query "Explain the methodology used in this paper"

# Retrieve more context
hybrid-rag query "Summarize the results" --top-k 10
```

---

## Configuration

All configuration is managed through environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `LLM_MODEL` | `phi3` | Language model for generation |
| `FAISS_INDEX_PATH` | `./faiss_index` | Directory for persistent FAISS index |

### Chunking Parameters

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `500` | Characters per chunk |
| `overlap` | `50` | Overlap characters between consecutive chunks |

---

## Project Structure

```
src/hybrid_rag/
├── domain/
│   ├── entities.py        # Document, Chunk
│   ├── value_objects.py   # EmbeddingVector, ChunkMetadata, RetrievalResult
│   ├── ports.py           # EmbeddingProvider, VectorStore, LanguageModel, DocumentReader, GraphStore
│   └── services.py        # chunk_text() — overlapping text splitter
├── application/
│   ├── dtos.py            # IngestResult, QueryResult
│   ├── ingest.py          # IngestDocumentUseCase
│   └── query.py           # QueryKnowledgeBaseUseCase
├── infrastructure/
│   ├── config.py           # Config dataclass (env-driven)
│   ├── ollama/
│   │   ├── embeddings.py   # OllamaEmbeddingProvider
│   │   └── llm.py          # OllamaLanguageModel
│   ├── faiss/
│   │   └── vector_store.py # FAISSVectorStore
│   └── pypdf/
│       └── reader.py       # PyPDFDocumentReader
└── presentation/
    └── cli.py              # Typer CLI (ingest, query)

tests/
└── test_use_cases.py       # Smoke tests with fake adapters
```

---

## Roadmap

- [ ] **Knowledge Graph Store** — NetworkX-backed `GraphStore` adapter with entity/relation extraction
- [ ] **Hybrid Retrieval Fusion** — Combine vector + graph scores with reciprocal rank fusion
- [ ] **Batch Embedding** — Parallel embedding for large document sets
- [ ] **Approximate FAISS Index** — IVF/HNSW for sub-linear search at scale
- [ ] **Sentence-Aware Chunking** — Split on sentence boundaries instead of fixed characters
- [ ] **Additional Adapters** — Qdrant, Chroma, OpenAI, Unstructured.io
- [ ] **API Server** — FastAPI/REST interface alongside CLI

---

## Running Tests

```bash
pytest tests/
```

Tests use fake adapter implementations — no Ollama, no FAISS, no PDFs required.

---

## License

Not yet licensed. Contact the author for usage rights.