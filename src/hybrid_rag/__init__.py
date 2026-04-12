"""Hybrid RAG — Retrieval-Augmented Generation with vector search and knowledge graph.

Layered architecture following Domain-Driven Design:

- **domain**      — entities, value objects, services, and port ABCs
- **application**  — use-case orchestration (depends only on domain)
- **infrastructure** — concrete adapters (Ollama, FAISS, PyPDF)
- **presentation**   — CLI entry point
"""