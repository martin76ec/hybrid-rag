"""Command-line interface for the hybrid-RAG project.

The CLI is built with **typer** and exposes two commands:

1. ``ingest`` – read one or more PDF files, chunk them, embed the text via the
   Ollama embedding model, and store the vectors in a FAISS index.
2. ``query`` – ask a natural-language question. The query is embedded, the most
   relevant chunks are retrieved from the FAISS index, and the LLM (also served
   by Ollama) generates an answer using those chunks as context.
"""

import sys
from pathlib import Path
from typing import List

import typer
from rich import print as rprint

from ..application.ingest import IngestDocumentUseCase
from ..application.query import QueryKnowledgeBaseUseCase
from ..infrastructure.config import Config
from ..infrastructure.faiss import FAISSVectorStore
from ..infrastructure.ollama import OllamaEmbeddingProvider, OllamaLanguageModel
from ..infrastructure.pypdf import PyPDFDocumentReader

app = typer.Typer(help="Hybrid Retrieval-Augmented Generation CLI")


def _wire_ingest_use_case() -> IngestDocumentUseCase:
    """Construct the ingest use case with all concrete dependencies."""
    cfg = Config()
    return IngestDocumentUseCase(
        reader=PyPDFDocumentReader(),
        embedder=OllamaEmbeddingProvider(cfg.ollama_host, cfg.embedding_model),
        store=FAISSVectorStore(cfg.faiss_index_path),
    )


def _wire_query_use_case() -> QueryKnowledgeBaseUseCase:
    """Construct the query use case with all concrete dependencies."""
    cfg = Config()
    return QueryKnowledgeBaseUseCase(
        embedder=OllamaEmbeddingProvider(cfg.ollama_host, cfg.embedding_model),
        store=FAISSVectorStore(cfg.faiss_index_path),
        llm=OllamaLanguageModel(cfg.ollama_host, cfg.llm_model),
    )


@app.command()
def ingest_cmd(
    pdf_paths: List[Path] = typer.Argument(..., help="Path(s) to PDF files to ingest"),
    chunk_size: int = typer.Option(500, "-c", "--chunk-size", help="Approximate characters per chunk"),
) -> None:
    """Ingest PDF(s) into the knowledge graph / vector store."""
    use_case = _wire_ingest_use_case()
    for pdf_path in pdf_paths:
        if not pdf_path.is_file():
            rprint(f"[red]File not found:[/red] {pdf_path}")
            continue
        rprint(f"[green]Processing:[/green] {pdf_path}")
        try:
            result = use_case.execute(str(pdf_path), chunk_size=chunk_size)
            rprint(f"[green]Ingested:[/green] {pdf_path} ({result.num_chunks} chunks)")
        except Exception as exc:
            rprint(f"[red]Error ingesting {pdf_path}: {exc}")
            sys.exit(1)


@app.command()
def query_cmd(
    question: str = typer.Argument(..., help="Natural language query"),
    top_k: int = typer.Option(5, "-k", "--top-k", help="Number of retrieved chunks to use"),
) -> None:
    """Ask a question against the indexed knowledge."""
    use_case = _wire_query_use_case()
    result = use_case.execute(question, top_k=top_k)
    rprint("[bold blue]Answer:[/bold blue]\n" + result.answer)
    if result.sources:
        rprint("[dim]Sources: " + ", ".join(result.sources) + "[/dim]")


if __name__ == "__main__":
    app()