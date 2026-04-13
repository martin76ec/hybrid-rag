"""NetworkX-based knowledge graph store — implements :class:`GraphStore`.

Stores triples as a directed multigraph in NetworkX and persists to JSON.
Supports neighbour-based retrieval: given query entities, expand to
neighbouring nodes and return the associated chunk metadata.

Uses vis-network.js (same rendering engine as Neo4j Browser) for graph
visualisation — rendered as standalone HTML with data embedded as JSON.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import networkx as nx

from ...domain.ports import GraphStore
from ...domain.value_objects import ChunkMetadata, RetrievalResult, Triple
from ..templates.renderer import render_graph_html

log = logging.getLogger(__name__)

PALETTE = [
    "#4cc9f0",
    "#f72585",
    "#7209b7",
    "#3a0ca3",
    "#4361ee",
    "#4895ef",
    "#560bad",
    "#480ca8",
]


class NetworkXGraphStore(GraphStore):
    """Concrete :class:`GraphStore` backed by a NetworkX directed multigraph."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._graph: nx.MultiDiGraph
        self._triple_sources: dict[int, Triple] = {}
        self._entity_to_chunks: dict[str, list[ChunkMetadata]] = {}
        self._entity_vectors: dict[str, list[float]] = {}
        self._load_or_create()

    def add_triples(self, triples: list[Triple]) -> None:
        for t in triples:
            self._graph.add_edge(t.subject, t.obj, predicate=t.predicate)
            edge_key = hash((t.subject, t.predicate, t.obj))
            self._triple_sources[edge_key] = t
            chunk_meta = ChunkMetadata(
                source=t.source, chunk_index=t.chunk_index, text=t.chunk_text
            )
            for entity in (t.subject, t.obj):
                if entity not in self._entity_to_chunks:
                    self._entity_to_chunks[entity] = []
                existing_texts = {c.text for c in self._entity_to_chunks[entity]}
                if chunk_meta.text not in existing_texts:
                    self._entity_to_chunks[entity].append(chunk_meta)
        self._persist()

    def query(self, question: str, top_k: int = 10) -> list[RetrievalResult]:
        entities = self._extract_entity_mentions(question)
        visited_nodes: set[str] = set()
        relevant_chunks: list[ChunkMetadata] = []

        for entity in entities:
            if entity not in self._graph:
                continue
            for neighbour in nx.single_source_shortest_path_length(
                self._graph, entity, cutoff=2
            ):
                if neighbour in visited_nodes:
                    continue
                visited_nodes.add(neighbour)
                for chunk in self._entity_to_chunks.get(neighbour, []):
                    if chunk not in relevant_chunks:
                        relevant_chunks.append(chunk)

        results = [
            RetrievalResult(chunk=c, score=1.0 / (1 + i))
            for i, c in enumerate(relevant_chunks[:top_k])
        ]
        return results

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def all_triples(self) -> list[Triple]:
        triples: list[Triple] = []
        for u, v, data in self._graph.edges(data=True):
            edge_key = hash((u, data.get("predicate", ""), v))
            t = self._triple_sources.get(edge_key)
            if t:
                triples.append(t)
            else:
                triples.append(
                    Triple(subject=u, predicate=data.get("predicate", ""), obj=v)
                )
        return triples

    def extract_entity_mentions(self, question: str) -> list[str]:
        return self._extract_entity_mentions(question)

    def set_entity_vectors(self, vectors: dict[str, list[float]]) -> None:
        self._entity_vectors.update(vectors)
        self._persist_vectors()

    def render_graph(
        self, output_path: str, matched_entities: list[str] | None = None
    ) -> str:
        """Render the graph as a vis-network.js HTML file (Neo4j Browser style).

        If *matched_entities* is provided, those nodes are highlighted in
        yellow and their 2-hop neighbours in orange.
        """
        source_colors: dict[str, str] = {}
        color_idx = 0
        matched_set = set(matched_entities or [])

        neighbour_set: set[str] = set()
        if matched_entities:
            ne = self._extract_entity_mentions(" ".join(matched_entities))
            for entity in ne:
                if entity in self._graph:
                    neighbour_set.update(
                        nx.single_source_shortest_path_length(
                            self._graph, entity, cutoff=2
                        )
                    )

        nodes = []
        for node_id in self._graph.nodes:
            chunks = self._entity_to_chunks.get(node_id, [])
            src = Path(chunks[0].source).stem if chunks else "unknown"
            if src not in source_colors:
                source_colors[src] = PALETTE[color_idx % len(PALETTE)]
                color_idx += 1

            if node_id in matched_set:
                color = "#ffd60a"
            elif node_id in neighbour_set:
                color = "#ff9e00"
            else:
                color = source_colors[src]

            title_parts = [f"<b>{node_id}</b><br>Source: {src}"]
            for c in chunks[:3]:
                title_parts.append(
                    f"Chunk {c.chunk_index}: {c.text[:80]}..."
                    if len(c.text) > 80
                    else f"Chunk {c.chunk_index}: {c.text}"
                )
            nodes.append(
                {
                    "id": node_id,
                    "label": node_id,
                    "color": color,
                    "title": "<br>".join(title_parts),
                    "_source": src,
                    "_neighbour": node_id in neighbour_set,
                }
            )

        highlight_edges: set[tuple[str, str]] = set()
        if matched_entities:
            for e in self._extract_entity_mentions(" ".join(matched_entities)):
                if e in self._graph:
                    for n in nx.single_source_shortest_path_length(
                        self._graph, e, cutoff=2
                    ):
                        highlight_edges.add((e, n))
                        highlight_edges.add((n, e))

        edges = []
        for u, v, data in self._graph.edges(data=True):
            edges.append(
                {
                    "from": u,
                    "to": v,
                    "label": data.get("predicate", ""),
                }
            )

        return render_graph_html(
            nodes, edges, output_path, highlighted_nodes=matched_entities
        )

    def _extract_entity_mentions(self, question: str) -> list[str]:
        regex_entities = self._regex_match(question)
        if self._entity_vectors:
            semantic_entities = self._semantic_match(question)
            seen = set(regex_entities)
            for e in semantic_entities:
                if e not in seen:
                    regex_entities.append(e)
                    seen.add(e)
        return regex_entities

    def _regex_match(self, question: str) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question)
        entities: list[str] = []
        matched: set[str] = set()
        for node in self._graph.nodes:
            for word in words:
                if word.lower() in node.lower() and node not in matched:
                    entities.append(node)
                    matched.add(node)
                    break
        return entities

    def _semantic_match(self, question: str) -> list[str]:
        if not self._entity_vectors:
            return []
        q_words = question.split()
        scored: list[tuple[float, str]] = []
        for entity, _vec in self._entity_vectors.items():
            entity_words = entity.lower().split()
            hits = sum(1 for w in q_words if w.lower() in entity_words)
            if hits == 0:
                continue
            score = hits / max(len(entity_words), 1)
            scored.append((score, entity))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:10]]

    def _persist(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph)
        (self._path / "graph.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        meta = {
            k: [
                {"source": c.source, "chunk_index": c.chunk_index, "text": c.text}
                for c in v
            ]
            for k, v in self._entity_to_chunks.items()
        }
        (self._path / "entity_chunks.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        self._persist_vectors()

    def _persist_vectors(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        if self._entity_vectors:
            (self._path / "entity_vectors.json").write_text(
                json.dumps(self._entity_vectors, ensure_ascii=False), encoding="utf-8"
            )

    def _load_or_create(self) -> None:
        graph_file = self._path / "graph.json"
        chunks_file = self._path / "entity_chunks.json"
        vectors_file = self._path / "entity_vectors.json"
        if graph_file.is_file():
            data = json.loads(graph_file.read_text(encoding="utf-8"))
            self._graph = nx.node_link_graph(data, directed=True, multigraph=True)
        else:
            self._graph = nx.MultiDiGraph()
        if chunks_file.is_file():
            raw = json.loads(chunks_file.read_text(encoding="utf-8"))
            self._entity_to_chunks = {
                k: [
                    ChunkMetadata(
                        source=c["source"], chunk_index=c["chunk_index"], text=c["text"]
                    )
                    for c in v
                ]
                for k, v in raw.items()
            }
        if vectors_file.is_file():
            self._entity_vectors = json.loads(vectors_file.read_text(encoding="utf-8"))
