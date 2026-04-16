"""Gradio web interface for Hybrid-RAG.

Three tabs: Ingest (upload PDFs and build graph), Query (ask questions with
visual retrieval provenance), and Graph (interactive knowledge-graph explorer).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import gradio as gr

from ..application.ingest import INGEST_STEPS, IngestDocumentUseCase
from ..application.query import QUERY_STEPS, QueryKnowledgeBaseUseCase
from ..infrastructure.config import Config
from ..infrastructure.faiss import FAISSVectorStore
from ..infrastructure.logging import setup_logging
from ..infrastructure.networkx import NetworkXGraphStore
from ..infrastructure.ollama import (
    OllamaClient,
    OllamaDocumentAnalyzer,
    OllamaEmbeddingProvider,
    OllamaLanguageModel,
    OllamaTripleExtractor,
    OllamaTripleRefiner,
)
from ..infrastructure.pypdf import PyPDFDocumentReader
from .stepper import render_stepper


def _wire() -> tuple[Config, OllamaClient, FAISSVectorStore, NetworkXGraphStore]:
    cfg = Config()
    setup_logging(cfg.log_level)
    client = OllamaClient(
        cfg.ollama_host,
        max_retries=cfg.ollama_max_retries,
        max_concurrency=cfg.ollama_max_concurrency,
        cache_dir=cfg.ollama_cache_dir,
    )
    store = FAISSVectorStore(cfg.faiss_index_path)
    graph = NetworkXGraphStore(cfg.graph_store_path)
    return cfg, client, store, graph


def _embed_graph_html(path: str) -> str:
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


def ingest_tab(files, chunk_size) -> Generator:
    cfg, client, store, graph = _wire()
    if not files:
        yield (render_stepper(INGEST_STEPS, -1), "No files uploaded.", "", "")
        return

    embedder = OllamaEmbeddingProvider(client, cfg.embedding_model)
    extractor = OllamaTripleExtractor(client, cfg.llm_model)
    refiner = OllamaTripleRefiner(client, cfg.llm_model)
    doc_analyzer = OllamaDocumentAnalyzer(client, cfg.llm_model)
    uc = IngestDocumentUseCase(
        reader=PyPDFDocumentReader(),
        embedder=embedder,
        store=store,
        triple_extractor=extractor,
        graph_store=graph,
        triple_refiner=refiner,
        document_analyzer=doc_analyzer,
    )

    total_chunks = 0
    total_triples = 0
    log_lines: list[str] = []
    extraction_log = ""

    for i, f in enumerate(files, 1):
        path = f.name if hasattr(f, "name") else str(f)
        name = Path(path).stem
        partial_log = f"Processing file {i}/{len(files)}…\n  ▶ {name}\n\n" + "\n".join(
            log_lines
        )
        yield (render_stepper(INGEST_STEPS, -1), partial_log, "", "")

        gen = uc.execute_stepped(path, chunk_size=int(chunk_size), overlap=60)
        result = None
        while True:
            try:
                step_idx, step_label = next(gen)
                progress_log = (
                    f"Processing file {i}/{len(files)}…  [{step_label}]\n\n"
                    + "\n".join(log_lines)
                )
                yield (render_stepper(INGEST_STEPS, step_idx), progress_log, "", "")
            except StopIteration as exc:
                result = exc.value
                break
            except Exception as exc:
                err_msg = f"❌ Error at step {step_label}: {exc}"
                log_lines.append(f"  ✗ {name}: {exc}")
                yield (render_stepper(INGEST_STEPS, step_idx), err_msg, "", "")
                return

        total_chunks += result.num_chunks
        total_triples += result.num_triples
        log_lines.append(
            f"  ✓ {name}: {result.num_chunks} chunks, {result.num_triples} triples"
        )
        extraction_log = ""
        if result.extraction_summary:
            s = result.extraction_summary
            parts = [f"--- {name} ---"]
            if s.doc_type:
                parts.append(f"Document type: {s.doc_type}")
            if s.doc_description:
                parts.append(f"Description: {s.doc_description}")
            if s.suggested_triple_patterns:
                parts.append(
                    f"Suggested triple patterns: {len(s.suggested_triple_patterns)}"
                )
                for p in s.suggested_triple_patterns:
                    parts.append(f"  - {p}")
            parts.extend(
                [
                    f"Raw triples extracted: {len(s.raw_triples)}",
                    f"Entities canonicalized: {len(s.canonical_mapping)}",
                    f"Predicates shortened: {len(s.shortened_predicates)}",
                    f"Trivial triples removed: {len(s.removed_triples)}",
                    f"Missing edges added: {len(s.added_triples)}",
                    f"Final triples in graph: {len(s.refined_triples)}",
                ]
            )
            extraction_log = "\n".join(parts)

        progress_log = f"Processing file {i}/{len(files)}…\n\n" + "\n".join(log_lines)
        if i < len(files):
            progress_log += f"\n\nNext: file {i + 1}/{len(files)}…"
        yield (
            render_stepper(INGEST_STEPS, len(INGEST_STEPS) - 1),
            progress_log,
            extraction_log,
            "",
        )

    ingest_log = (
        f"✅ Ingested {len(files)} file(s)\n"
        f"Total chunks: {total_chunks}\n"
        f"Total triples: {total_triples}\n\n" + "\n".join(log_lines)
    )

    graph = NetworkXGraphStore(cfg.graph_store_path)
    html_path = os.path.join(cfg.graph_store_path, "knowledge_graph.html")
    graph.render_graph(html_path)

    yield (
        render_stepper(INGEST_STEPS, len(INGEST_STEPS) - 1),
        ingest_log,
        extraction_log,
        _embed_graph_html(html_path),
    )


def query_tab(question, top_k) -> Generator:
    cfg, client, store, graph = _wire()
    embedder = OllamaEmbeddingProvider(client, cfg.embedding_model)
    llm = OllamaLanguageModel(client, cfg.llm_model)

    uc = QueryKnowledgeBaseUseCase(
        embedder=embedder,
        store=store,
        llm=llm,
        graph_store=graph,
    )

    yield (render_stepper(QUERY_STEPS, -1), "", None, [], "")

    gen = uc.execute_verbose_stepped(question, top_k=int(top_k))
    result = None
    step_label = ""
    step_idx = -1
    while True:
        try:
            step_idx, step_label = next(gen)
            yield (
                render_stepper(QUERY_STEPS, step_idx),
                f"[{step_label}]…",
                None,
                [],
                "",
            )
        except StopIteration as exc:
            result = exc.value
            break
        except Exception as exc:
            err_msg = f"❌ Error at step {step_label}: {exc}"
            yield (render_stepper(QUERY_STEPS, step_idx), err_msg, None, [], "")
            return

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

    yield (
        render_stepper(QUERY_STEPS, len(QUERY_STEPS) - 1),
        result.answer,
        _embed_graph_html(html_path),
        rows,
        entities_text,
    )


def graph_tab():
    cfg, _, _, graph = _wire()
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
                ingest_stepper = gr.HTML(
                    value=render_stepper(INGEST_STEPS, -1),
                    label="Pipeline progress",
                )
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
                ingest_log = gr.Textbox(label="Ingest log", lines=12, interactive=False)
                extraction_log = gr.Textbox(
                    label="Extraction pipeline", lines=10, interactive=False
                )
                ingest_graph = gr.HTML(label="Knowledge graph")

                ingest_btn.click(
                    fn=ingest_tab,
                    inputs=[files, chunk_size],
                    outputs=[ingest_stepper, ingest_log, extraction_log, ingest_graph],
                )

            with gr.Tab("Query"):
                query_stepper = gr.HTML(
                    value=render_stepper(QUERY_STEPS, -1),
                    label="Pipeline progress",
                )
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
                    outputs=[
                        query_stepper,
                        answer,
                        query_graph,
                        results_table,
                        matched,
                    ],
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
