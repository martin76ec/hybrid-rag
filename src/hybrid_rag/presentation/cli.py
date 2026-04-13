"""Command-line interface for the hybrid-RAG project.

The CLI is built with **typer** and exposes four commands:

1. ``ingest`` – read PDFs, chunk text, embed via Ollama, store vectors in
   FAISS, extract knowledge-graph triples via LLM, and store in NetworkX.
2. ``query`` – ask a question; vector + graph results are fused with RRF
   and passed to the LLM for answer generation.
3. ``graph`` – inspect the knowledge graph (node/edge counts, triple dump,
   DOT export, or interactive HTML visualisation).
4. ``web`` – launch the Gradio web UI with ingest, query (with retrieval
   provenance), and graph explorer tabs.
"""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rprint
from rich.table import Table

from ..application.ingest import IngestDocumentUseCase
from ..application.query import QueryKnowledgeBaseUseCase
from ..infrastructure.config import Config
from ..infrastructure.faiss import FAISSVectorStore
from ..infrastructure.networkx import NetworkXGraphStore
from ..infrastructure.ollama import (
    OllamaEmbeddingProvider,
    OllamaLanguageModel,
    OllamaTripleExtractor,
    OllamaTripleRefiner,
)
from ..infrastructure.pypdf import PyPDFDocumentReader

app = typer.Typer(help="Hybrid Retrieval-Augmented Generation CLI")


def _cfg() -> Config:
    return Config()


def _wire_ingest_use_case() -> IngestDocumentUseCase:
    """Construct the ingest use case with all concrete dependencies."""
    cfg = _cfg()
    triple_extractor = OllamaTripleExtractor(cfg.ollama_host, cfg.llm_model)
    triple_refiner = OllamaTripleRefiner(cfg.ollama_host, cfg.llm_model)
    graph_store = NetworkXGraphStore(cfg.graph_store_path)
    embedder = OllamaEmbeddingProvider(cfg.ollama_host, cfg.embedding_model)
    return IngestDocumentUseCase(
        reader=PyPDFDocumentReader(),
        embedder=embedder,
        store=FAISSVectorStore(cfg.faiss_index_path),
        triple_extractor=triple_extractor,
        graph_store=graph_store,
        triple_refiner=triple_refiner,
    )


def _wire_query_use_case() -> QueryKnowledgeBaseUseCase:
    """Construct the query use case with all concrete dependencies."""
    cfg = _cfg()
    graph_store = NetworkXGraphStore(cfg.graph_store_path)
    return QueryKnowledgeBaseUseCase(
        embedder=OllamaEmbeddingProvider(cfg.ollama_host, cfg.embedding_model),
        store=FAISSVectorStore(cfg.faiss_index_path),
        llm=OllamaLanguageModel(cfg.ollama_host, cfg.llm_model),
        graph_store=graph_store,
    )


@app.command()
def ingest_cmd(
    pdf_paths: List[Path] = typer.Argument(..., help="Path(s) to PDF files to ingest"),
    chunk_size: int = typer.Option(
        500, "-c", "--chunk-size", help="Approximate characters per chunk"
    ),
) -> None:
    """Ingest PDF(s) into the vector store and knowledge graph."""
    use_case = _wire_ingest_use_case()
    for pdf_path in pdf_paths:
        if not pdf_path.is_file():
            rprint(f"[red]File not found:[/red] {pdf_path}")
            continue
        rprint(f"[green]Processing:[/green] {pdf_path}")
        try:
            result = use_case.execute(str(pdf_path), chunk_size=chunk_size)
            rprint(
                f"[green]Ingested:[/green] {pdf_path} "
                f"({result.num_chunks} chunks, {result.num_triples} triples)"
            )
            if result.extraction_summary:
                s = result.extraction_summary
                rprint(f"  [dim]Raw triples:[/dim] {len(s.raw_triples)}")
                if s.canonical_mapping:
                    rprint(
                        f"  [dim]Canonical entities:[/dim] {len(s.canonical_mapping)} merged"
                    )
                if s.shortened_predicates:
                    rprint(
                        f"  [dim]Shortened predicates:[/dim] {len(s.shortened_predicates)}"
                    )
                if s.removed_triples:
                    rprint(f"  [dim]Removed trivial:[/dim] {len(s.removed_triples)}")
                if s.added_triples:
                    rprint(f"  [dim]Added missing:[/dim] {len(s.added_triples)}")
                rprint(f"  [dim]Final triples:[/dim] {len(s.refined_triples)}")
        except Exception as exc:
            rprint(f"[red]Error ingesting {pdf_path}: {exc}")
            sys.exit(1)


@app.command()
def query_cmd(
    question: str = typer.Argument(..., help="Natural language query"),
    top_k: int = typer.Option(
        5, "-k", "--top-k", help="Number of retrieved chunks to use"
    ),
) -> None:
    """Ask a question against the indexed knowledge."""
    use_case = _wire_query_use_case()
    result = use_case.execute(question, top_k=top_k)
    rprint("[bold blue]Answer:[/bold blue]\n" + result.answer)
    if result.sources:
        rprint("[dim]Sources: " + ", ".join(result.sources) + "[/dim]")


@app.command(name="graph")
def graph_cmd(
    format: str = typer.Option(
        "summary", "-f", "--format", help="Output format: summary, triples, dot, html"
    ),
) -> None:
    """Inspect the knowledge graph."""
    cfg = _cfg()
    store = NetworkXGraphStore(cfg.graph_store_path)
    triples = store.all_triples()

    if format == "summary":
        table = Table(title="Knowledge Graph Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Nodes", str(store.node_count()))
        table.add_row("Edges", str(store.edge_count()))
        rprint(table)

    elif format == "triples":
        for t in triples:
            rprint(
                f"[cyan]{t.subject}[/cyan] — [yellow]{t.predicate}[/yellow] → [magenta]{t.obj}[/magenta]  [dim]({t.source})[/dim]"
            )

    elif format == "dot":
        lines = ["digraph knowledge_graph {"]
        for t in triples:
            s = t.subject.replace('"', '\\"')
            p = t.predicate.replace('"', '\\"')
            o = t.obj.replace('"', '\\"')
            lines.append(f'  "{s}" -> "{o}" [label="{p}"];')
        lines.append("}")
        rprint("\n".join(lines))

    elif format == "html":
        output = str(Path(cfg.graph_store_path) / "knowledge_graph.html")
        store.render_graph(output)
        rprint(f"[green]Interactive graph written to:[/green] {output}")
        rprint("[dim]Open in a browser to explore.[/dim]")

    else:
        rprint(
            f"[red]Unknown format:[/red] {format}. Use summary, triples, dot, or html."
        )


@app.command(name="web")
def web_cmd(
    port: int = typer.Option(7860, "-p", "--port", help="Port to serve the UI on"),
) -> None:
    """Launch the Gradio web interface."""
    from ..presentation.web import build_app

    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    app()
