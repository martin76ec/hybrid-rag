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
| Knowledge Graph (NetworkX) | Structured relationships, multi-hop queries | Vague or paraphrased concepts |

Hybrid-RAG runs both and fuses the results via **Reciprocal Rank Fusion** — so you get the best of both worlds.

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
│   LLM    │          │          │  Triple    │
│          │          │          │  Extractor │
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
- **Knowledge Graph** — LLM-extracted entity-relation triples stored in a NetworkX directed multigraph with JSON persistence
- **Hybrid Retrieval** — Vector search + graph neighbour traversal fused via Reciprocal Rank Fusion (RRF)
- **LLM-Powered Q&A** — Generate grounded answers from fused context via Ollama (default: `llama3.2`)
- **Source Attribution** — Every answer includes the source chunks used to generate it
- **CLI Interface** — Beautiful terminal output with `rich` formatting, including a `graph` inspect command
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
ollama pull llama3.2
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
# Ingest PDFs into the vector store and knowledge graph
hybrid-rag ingest ./papers/attention-is-all-you-need.pdf

# Ask questions (uses hybrid retrieval: vector + graph via RRF)
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
# Basic query (returns top 5 fused chunks as context)
hybrid-rag query "Explain the methodology used in this paper"

# Retrieve more context
hybrid-rag query "Summarize the results" --top-k 10
```

### Knowledge Graph

```bash
# Summary (node and edge counts)
hybrid-rag graph

# Dump all triples
hybrid-rag graph --format triples

# Export as DOT (pipe to Graphviz for visualisation)
hybrid-rag graph --format dot | dot -Tpng -o graph.png

# Interactive HTML (open in browser — zoom, pan, drag nodes)
hybrid-rag graph --format html
```

### Web UI

```bash
# Launch the Gradio interface on port 7860
hybrid-rag web

# Custom port
hybrid-rag web --port 8080
```

The web UI has three tabs:

| Tab | What It Shows |
|---|---|
| **Ingest** | Upload PDFs, set chunk size, see progress log and the resulting knowledge graph |
| **Query** | Ask questions — shows the LLM answer, a highlighted graph (matched entities in yellow, 2-hop neighbours in orange), an RRF fusion table with per-chunk scores and which path contributed, and which graph entities were matched |
| **Graph** | Full interactive knowledge-graph explorer with refresh |

---

## Configuration

All configuration is managed through environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `LLM_MODEL` | `llama3.2` | Language model for generation and triple extraction |
| `FAISS_INDEX_PATH` | `./faiss_index` | Directory for persistent FAISS index |
| `GRAPH_STORE_PATH` | `./graph_store` | Directory for persistent knowledge graph |

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
│   ├── value_objects.py   # EmbeddingVector, ChunkMetadata, RetrievalResult, Triple
│   ├── ports.py           # EmbeddingProvider, VectorStore, LanguageModel, DocumentReader, GraphStore, TripleExtractor
│   └── services.py        # chunk_text(), reciprocal_rank_fusion()
├── application/
│   ├── dtos.py            # IngestResult, QueryResult
│   ├── ingest.py          # IngestDocumentUseCase
│   └── query.py           # QueryKnowledgeBaseUseCase
├── infrastructure/
│   ├── config.py           # Config dataclass (env-driven)
│   ├── ollama/
│   │   ├── embeddings.py   # OllamaEmbeddingProvider
│   │   ├── llm.py          # OllamaLanguageModel
│   │   └── triple_extractor.py  # OllamaTripleExtractor
│   ├── faiss/
│   │   └── vector_store.py # FAISSVectorStore
│   ├── networkx/
│   │   └── graph_store.py  # NetworkXGraphStore
│   └── pypdf/
│       └── reader.py       # PyPDFDocumentReader
└── presentation/
    ├── cli.py              # Typer CLI (ingest, query, graph, web)
    └── web.py              # Gradio web UI

tests/
├── test_use_cases.py       # Smoke tests with fake adapters
├── test_e2e.py             # End-to-end quality metrics
└── fixtures/
    └── pdf_factory.py      # Synthetic test PDF generator
```

---

## Roadmap

- [ ] **Batch Embedding** — Parallel embedding for large document sets
- [ ] **Approximate FAISS Index** — IVF/HNSW for sub-linear search at scale
- [ ] **Sentence-Aware Chunking** — Split on sentence boundaries instead of fixed characters
- [ ] **Additional Adapters** — Qdrant, Chroma, OpenAI, Unstructured.io
- [ ] **API Server** — FastAPI/REST interface alongside CLI

---

## Running Tests

```bash
# Unit tests (no external services needed)
pytest tests/test_use_cases.py

# E2E tests (requires Ollama running with nomic-embed-text + llama3.2)
pytest tests/test_e2e.py -v
```

---

## E2E Quality Metrics

The E2E tests use two **synthetic PDFs** with deliberately planted facts — every entity name, numeric value, and relationship is known in advance, so we can measure exactly what the pipeline extracts, retrieves, and generates. All examples below are from an **actual pipeline run** (Ollama + FAISS + NetworkX), not hypothetical.

### Test Corpus

**`tech_company.pdf`** — a fictional annual report for NovaMind Technologies:

| Category | Fact Planted in PDF |
|---|---|
| Founders | Dr. Elena Vasquez, Raj Patel |
| HQ | Austin, Texas |
| Product | Cortex-7 processor |
| Performance | 120 tera-ops/sec, 15 watts, 8x more efficient than competitors |
| Revenue FY2024 | $340 million (62% YoY increase) |
| Employees | 1,250 across Austin, Berlin, Singapore |
| CTO | Raj Patel; Board chair: General Marcus Webb (retired) |
| R&D | 47 papers, 89 patents, $95M budget (2025) |
| Manufacturing | Dresden, Germany; 5nm process; 94.2% yield; 1,800 die sites/wafer |
| Price | $4,200 per unit |
| Partners | Quantum Dynamics Ltd, MIT, Heidelberg University, ETH Zurich |
| IPO | Q3 2025, target valuation $5.8 billion |

**`clinical_trial.pdf`** — a fictional Phase III trial (VXC-204):

| Category | Fact Planted in PDF |
|---|---|
| Drug | Veratralimab 300mg IV every 4 weeks |
| Indication | Advanced pulmonary sarcoidosis (Stage III/IV) |
| Principal Investigator | Dr. Amara Okafor, MD, PhD, University of Michigan |
| Sponsor | Veritas Biopharma Inc. |
| Enrollment | 648 participants, 42 sites, 9 countries |
| Primary endpoint | FVC % predicted change at week 52: +6.8pp vs -0.4 placebo (p < 0.001) |
| Secondary endpoint | SGRQ total score: -7.5 vs -1.2 placebo (p = 0.003) |
| Responder rate | 58% (treatment) vs 19% (placebo) achieved ≥5% FVC improvement |
| Top adverse events | URTI 12.3%, fatigue 9.8%, infusion reactions 7.1% |
| Serious AEs | Pneumocystis pneumonia (1), grade 3 hepatotoxicity (2) |
| Pharmacokinetics | Half-life 18.6 days, steady state by week 12 |
| Subgroup | HLA-DRB1\*04 allele → +9.3pp FVC improvement |
| Regulatory | FDA BLA Q1 2025, EMA Q2 2025 |

---

### How the Pipeline Processes a Test PDF

After text extraction, the PDF is split into overlapping chunks (400 chars, 60-char overlap). Here is **chunk 0 from the actual run**:

> *NovaMind Technologies Annual Report 2024*
> *NovaMind Technologies was founded in 2018 by Dr. Elena Vasquez and Raj Patel in Austin, Texas. The company specializes in neuromorphic computing chips that mimic biological neural networks. Their flagship product, the Cortex-7 processor, achieves 120 tera-operations per second while consuming only 15 watts of power, which is 8x more efficient than competing*

**Step 1 — Embedding (vector path):** The chunk text is sent to `nomic-embed-text` via Ollama, producing a 768-dim float vector. This vector + the chunk metadata `(source, chunk_index, text)` are written to the FAISS `IndexFlatL2` and the `metadata.jsonl` sidecar. The full ingest produced **6 chunks** for the tech company PDF and **7 chunks** for the clinical trial PDF.

**Step 2 — Triple extraction (graph path):** The same chunk text is sent to the LLM with this prompt:

```
Extract knowledge-graph triples from the following text.
Return ONLY a JSON array of objects with keys "subject", "predicate", "object".
Each value must be a non-null string.

Text:
NovaMind Technologies was founded in 2018 by Dr. Elena Vasquez and Raj Patel...
```

Here are the **actual triples the LLM extracted** across all chunks (25 total from both PDFs):

```
TECH COMPANY PDF (15 triples, 36 nodes, 25 edges):
  (Marcus Webb,               was chaired by,        the board of directors)
  (Dr. Yuki Tanaka,           led,                  NovaMind's research division)
  (NovaMind's research division, published,         47 peer-reviewed papers in 2024...)
  (NovaMind's research division, holds,              89 patents globally)
  (NovaMind's research division, has,                23 additional patents pending)
  (NovaMind,                  has R&D budget,         $95 million)
  (NovaMind,                  had R&D budget,         $72 million)
  (Cortex-7 processor,        is manufactured at,    NovaMind's fabrication facility in Dresden)
  (EuroChip Foundries,        licensed process node to, NovaMind)
  (NovaMind,                  part of,               European Neuromorphic Computing Consortium)
  (NovaMind,                  part of,               Heidelberg University)
  (NovaMind,                  part of,               ETH Zurich)
  (NovaMind,                  planned IPO,           Q3 2025)
  (NovaMind,                  target valuation,      $5.8 billion)

CLINICAL TRIAL PDF (10 triples):
  (VXC-204,                   Study Protocol,        A Phase III RCT of Veratralimab...)
  (adverse events...,         were characterized by,  upper respiratory tract infection (12.3%))
  (adverse events...,         were characterized by,  fatigue (9.8%))
  (adverse events...,         were characterized by,  mild infusion reactions (7.1%))
  (Three patients,            due to,                serious adverse events)
  (Veratralimab,              be approved by,        the FDA)
  (Veratralimab,              be indicated for,      advanced pulmonary sarcoidosis)
  (Veratralimab,              be approved to,        the EMA)
```

Every triple is enriched with `source`, `chunk_index`, and `chunk_text` from the originating chunk, so the graph can trace back to the exact passage.

**Step 3 — Graph construction:** `NetworkXGraphStore.add_triples()` does two things simultaneously:

1. **Creates edges** in a directed multigraph: `NovaMind ──planned IPO──▶ Q3 2025`, `Cortex-7 processor ──is manufactured at──▶ NovaMind's fabrication facility in Dresden`, etc.
2. **Builds the entity→chunks index**: maps each entity node to the `ChunkMetadata` of the chunk that mentioned it. From the actual run:

| Entity Node | Points to Chunk |
|---|---|
| `"NovaMind"` | `tech_company.pdf` chunk 3 |
| `"Cortex-7 processor"` | `tech_company.pdf` chunk 3 |
| `"Dr. Yuki Tanaka"` | `tech_company.pdf` chunk 2 |
| `"Marcus Webb"` | `tech_company.pdf` chunk 2 |
| `"Veratralimab"` | `clinical_trial.pdf` chunks 4, 5, 6 |

This index is the **bridge between the graph and the vector store** — it's what lets the graph return `RetrievalResult` objects with the same `ChunkMetadata` format that FAISS returns, enabling fusion downstream.

Each chunk therefore creates **two parallel representations**: a dense vector in FAISS and a set of symbolic edges in NetworkX.

---

### How a Query Traverses Both Stores

When the user asks **"Who founded NovaMind Technologies?"**:

```
┌─────────────────────────────────────────────────────────┐
│                     USER QUERY                           │
│          "Who founded NovaMind Technologies?"             │
└───────────────┬─────────────────────┬───────────────────┘
                │                     │
        ┌───────▼───────┐     ┌───────▼────────┐
        │  VECTOR PATH  │     │   GRAPH PATH   │
        │               │     │                │
        │  embed query  │     │  regex tokenize│
        │  via Ollama   │     │  ["Who",       │
        │      ↓        │     │   "founded",   │
        │  FAISS L2     │     │   "NovaMind"]  │
        │  search       │     │      ↓         │
        │      ↓        │     │  match nodes   │
        │  top-5 chunks │     │  containing    │
        │  ranked by    │     │  "NovaMind" →  │
        │  distance     │     │  "NovaMind"    │
        │               │     │      ↓         │
        │               │     │  2-hop BFS     │
        │               │     │  traversal     │
        │               │     │      ↓         │
        │               │     │  entity→chunks │
        │               │     │  index:        │
        │               │     │  NovaMind →    │
        │               │     │  chunk 3       │
        └───────┬───────┘     └───────┬────────┘
                │                     │
        ┌───────▼─────────────────────▼───────┐
        │       RECIPROCAL RANK FUSION         │
        │                                      │
        │  For each chunk (identified by       │
        │  source + chunk_index):              │
        │    score = Σ 1/(60 + rank)           │
        │    across both ranked lists          │
        │                                      │
        │  Chunks in BOTH lists get a          │
        │  boosted score → promoted            │
        └──────────────────┬───────────────────┘
                           │
                    ┌──────▼──────┐
                    │  top-k      │
                    │  fused      │
                    │  chunks     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  LLM prompt │
                    │  context +  │
                    │  question   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   ANSWER    │
                    └─────────────┘
```

**Vector path** — the question embedding is compared against all chunk vectors via L2 distance. This catches semantic similarity even when the question uses different words than the source text.

**Graph path** — the question is tokenized (`"NovaMind"` → regex match against graph nodes → hits `"NovaMind"` in the graph). Then a 2-hop BFS finds neighbours (e.g. `"$95 million"` at 1 hop, `"European Neuromorphic Computing Consortium"` at 1 hop). The `entity_to_chunks` index retrieves the originating chunk text for each matched/visited node.

**RRF fusion** — both paths produce ranked lists of `(ChunkMetadata, score)`. A chunk appearing in **both** lists accumulates `1/(60 + rank_vector) + 1/(60 + rank_graph)`, giving it a higher fused score than any chunk from a single list. Chunks that are both semantically similar *and* connected to the query entities get priority.

---

### What Each Metric Measures

#### 1. Retrieval Precision @ K (5 tests)

> Does the retrieval pipeline find chunks from the **right PDF**?

Each question targets one specific document. For example, *"What is the power consumption of the Cortex-7 processor?"* should return chunks from `tech_company.pdf`, not `clinical_trial.pdf`. This validates that the embedding space correctly distinguishes between two very different domains.

**Actual answer**: *"The Cortex-7 processor consumes 15 watts of power."* — sourced from `tech_company.pdf`.

**Pass condition**: the correct source filename appears in the top-5 fused results.

#### 2. Answer Keyword Overlap (5 tests)

> Does the LLM answer mention the **specific entities and facts** from the source?

The question *"Who founded NovaMind Technologies?"* has expected keywords `["Elena Vasquez", "Raj Patel", "Austin"]`.

**Actual answer**: *"Dr. Elena Vasquez and Raj Patel co-founded NovaMind Technologies in 2018."* — 2 of 3 keywords matched (67% overlap).

An answer like *"NovaMind was founded by its current leadership team"* would score 0% — grammatically correct but factually empty.

**Pass condition**: ≥ 30% of expected keywords appear in the answer.

#### 3. Answer Groundedness (2 tests)

> Does the answer state the **exact numeric facts** and avoid **planted wrong numbers**?

| Question | Ground Truth | Planted Negatives |
|---|---|---|
| *"How much revenue did NovaMind report in FY2024?"* | `"$340 million"` | `"$500 million"`, `"$200 million"`, `"$1 billion"` |
| *"How many participants were enrolled in VXC-204?"* | `"648"` | `"1000"`, `"500"`, `"2000"` |

**Actual answers**:
- *"NovaMind reported $340 million in revenue for fiscal year 2024, representing a 62% year-over-year increase."* — correct number present, no planted negatives.
- *"648 participants were enrolled in the VXC-204 clinical trial."* — correct number present, no planted negatives.

The planted negatives are plausible but wrong — if the LLM outputs one, it's hallucinating rather than reading context. This catches the most dangerous failure mode: a confident, specific, but fabricated statistic.

**Pass condition**: ground-truth number appears in answer; none of the planted negatives appear.

#### 4. Knowledge Graph Quality (5 tests)

> Does the triple extractor build a real graph, and does it contribute to retrieval?

| Test | What It Checks |
|---|---|
| Graph has nodes | Extraction didn't fail silently — actual count: **36 nodes** |
| Graph has edges | At least some relationships extracted — actual count: **25 edges** |
| Graph persists | `graph.json` and `entity_chunks.json` are written to disk |
| Known entities in triples | At least 1 of `NovaMind`, `Cortex-7`, `VXC-204`, `Veratralimab` appears — `NovaMind` and `Veratralimab` both present |
| Graph enriches retrieval | Querying `"NovaMind Cortex-7 processor"` returns chunks via 2-hop neighbour traversal — entity matching hits `NovaMind` and `Cortex-7 processor` nodes, both of which map to `tech_company.pdf` chunk 3 |

---

### Why These Four Dimensions?

They cover the full RAG quality stack end-to-end:

| Dimension | Catches |
|---|---|
| Retrieval Precision | Wrong chunks retrieved → answer built on irrelevant context |
| Keyword Overlap | LLM ignoring context → generic or evasive answers |
| Groundedness | Hallucination → confident but fabricated numbers |
| Graph Quality | Graph extraction failed or disconnected → hybrid adds no value over vector-only |

Each metric is computed against deterministic test data (the planted facts above), so regressions are caught immediately without ambiguity.

---

### Current Results (24/24 passing)

| Metric | Tests | Status |
|---|---|---|
| Retrieval Precision @ K | 5 | All pass |
| Answer Keyword Overlap | 5 | All pass (>= 30%) |
| Answer Groundedness | 2 | All pass (correct numbers, no hallucinations) |
| Knowledge Graph | 5 | All pass (36 nodes, 25 edges, persistence, entities, retrieval) |
| Ingestion Health | 3 | All pass (PDFs, index, metadata) |
| Unit Smoke Tests | 4 | All pass (fake adapters) |

---

## License

Not yet licensed. Contact the author for usage rights.