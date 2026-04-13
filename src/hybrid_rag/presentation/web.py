"""Gradio web interface for Hybrid-RAG.

Three tabs: Ingest (upload PDFs and build graph), Query (ask questions with
visual retrieval provenance), and Graph (interactive knowledge-graph explorer).
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

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


def _wire() -> tuple[Config, FAISSVectorStore, NetworkXGraphStore]:
    cfg = Config()
    store = FAISSVectorStore(cfg.faiss_index_path)
    graph = NetworkXGraphStore(cfg.graph_store_path)
    return cfg, store, graph


def _embed_graph_html(path: str) -> str:
    """Read a standalone HTML file and embed it directly in Gradio.

    The vis-network template is self-contained (loads vis.js from CDN),
    so we can inline it in gr.HTML with an iframe using a data URI.
    """
    if not os.path.isfile(path):
        return "<p>No graph data yet. Ingest some documents first.</p>"
    html_content = open(path, encoding="utf-8").read()
    import base64

    data_uri = "data:text/html;base64," + base64.b64encode(
        html_content.encode("utf-8")
    ).decode("ascii")
    return (
        f'<iframe src="{data_uri}" '
        f'style="width:100%; height:600px; border:none;" sandbox="allow-scripts allow-same-origin"></iframe>'
    )


def ingest_tab(files, chunk_size, progress=gr.Progress()):
    cfg, store, graph = _wire()
    if not files:
        return "No files uploaded.", "", ""

    embedder = OllamaEmbeddingProvider(cfg.ollama_host, cfg.embedding_model)
    extractor = OllamaTripleExtractor(cfg.ollama_host, cfg.llm_model)
    refiner = OllamaTripleRefiner(cfg.ollama_host, cfg.llm_model)
    uc = IngestDocumentUseCase(
        reader=PyPDFDocumentReader(),
        embedder=embedder,
        store=store,
        triple_extractor=extractor,
        graph_store=graph,
        triple_refiner=refiner,
    )

    total_chunks = 0
    total_triples = 0
    log_lines: list[str] = []
    last_summary = None

    for i, f in enumerate(files, 1):
        path = f.name if hasattr(f, "name") else str(f)
        progress(i / len(files), desc=f"Processing file {i}/{len(files)}")
        result = uc.execute(path, chunk_size=int(chunk_size), overlap=60)
        total_chunks += result.num_chunks
        total_triples += result.num_triples
        name = Path(path).stem
        log_lines.append(
            f"  {name}: {result.num_chunks} chunks, {result.num_triples} triples"
        )
        if result.extraction_summary:
            last_summary = result.extraction_summary

    ingest_log = (
        f"Ingested {len(files)} file(s)\n"
        f"Total chunks: {total_chunks}\n"
        f"Total triples: {total_triples}\n\n" + "\n".join(log_lines)
    )

    extraction_log = ""
    if last_summary:
        s = last_summary
        extraction_log = (
            f"Raw triples extracted: {len(s.raw_triples)}\n"
            f"Entities canonicalized: {len(s.canonical_mapping)}\n"
            f"Predicates shortened: {len(s.shortened_predicates)}\n"
            f"Trivial triples removed: {len(s.removed_triples)}\n"
            f"Missing edges added: {len(s.added_triples)}\n"
            f"Final triples in graph: {len(s.refined_triples)}"
        )

    graph = NetworkXGraphStore(cfg.graph_store_path)
    html_path = os.path.join(cfg.graph_store_path, "knowledge_graph.html")
    graph.render_graph(html_path)

    return ingest_log, extraction_log, _embed_graph_html(html_path)


def query_tab(question, top_k, progress=gr.Progress()):
    cfg, store, graph = _wire()
    embedder = OllamaEmbeddingProvider(cfg.ollama_host, cfg.embedding_model)
    llm = OllamaLanguageModel(cfg.ollama_host, cfg.llm_model)

    uc = QueryKnowledgeBaseUseCase(
        embedder=embedder,
        store=store,
        llm=llm,
        graph_store=graph,
    )

    progress(0.3, desc="Searching vector store and knowledge graph...")
    result = uc.execute_verbose(question, top_k=int(top_k))

    progress(0.8, desc="Generating highlighted graph...")
    html_path = os.path.join(cfg.graph_store_path, "query_graph.html")
    graph.render_graph(html_path, matched_entities=result.graph_entities)

    rows = []
    for chunk in result.fused_results:
        source_name = Path(chunk.source).stem if chunk.source else "?"
        text_preview = chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
        rows.append(
            [
                source_name,
                chunk.chunk_index,
                chunk.rrf_score,
                chunk.vector_rank or "-",
                chunk.graph_rank or "-",
                chunk.path,
                text_preview,
            ]
        )

    entities_text = (
        ", ".join(result.graph_entities) if result.graph_entities else "(none matched)"
    )

    return result.answer, _embed_graph_html(html_path), rows, entities_text


def graph_tab():
    cfg, _, graph = _wire()
    html_path = os.path.join(cfg.graph_store_path, "knowledge_graph.html")
    graph.render_graph(html_path)
    node_count = graph.node_count()
    edge_count = graph.edge_count()
    return _embed_graph_html(html_path), f"Nodes: {node_count}  |  Edges: {edge_count}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Hybrid-RAG") as app:
        gr.Markdown("# Hybrid-RAG\nVector search + knowledge graph, fused via RRF")

        with gr.Tabs():
            with gr.Tab("Ingest"):
                with gr.Row():
                    files = gr.File(
                        label="Upload PDFs",
                        file_count="multiple",
                        file_types=[".pdf"],
                    )
                chunk_size = gr.Slider(
                    100, 1000, value=400, step=50, label="Chunk size (chars)"
                )
                ingest_btn = gr.Button("Ingest", variant="primary")
                ingest_log = gr.Textbox(label="Ingest log", lines=8, interactive=False)
                extraction_log = gr.Textbox(
                    label="Extraction pipeline", lines=6, interactive=False
                )
                ingest_graph = gr.HTML(label="Knowledge graph")

                ingest_btn.click(
                    fn=ingest_tab,
                    inputs=[files, chunk_size],
                    outputs=[ingest_log, extraction_log, ingest_graph],
                )

            with gr.Tab("Query"):
                question = gr.Textbox(
                    label="Question",
                    placeholder="Ask something about your documents...",
                )
                top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K results")
                query_btn = gr.Button("Query", variant="primary")
                answer = gr.Textbox(label="Answer", lines=4, interactive=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        matched = gr.Textbox(
                            label="Graph entities matched", interactive=False
                        )
                    with gr.Column(scale=5):
                        pass
                query_graph = gr.HTML(label="Query graph (highlighted)")
                results_table = gr.Dataframe(
                    headers=[
                        "Source",
                        "Chunk",
                        "RRF Score",
                        "Vec Rank",
                        "Graph Rank",
                        "Path",
                        "Text",
                    ],
                    label="Fused results",
                    interactive=False,
                )

                query_btn.click(
                    fn=query_tab,
                    inputs=[question, top_k],
                    outputs=[answer, query_graph, results_table, matched],
                )

            with gr.Tab("Graph"):
                refresh_btn = gr.Button("Refresh graph")
                graph_stats = gr.Textbox(label="Graph stats", interactive=False)
                graph_html = gr.HTML(label="Knowledge graph")
                refresh_btn.click(
                    fn=graph_tab,
                    inputs=[],
                    outputs=[graph_html, graph_stats],
                )

    return app


def main():
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
