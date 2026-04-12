"""NetworkX-based knowledge graph store — implements :class:`GraphStore`.

Stores triples as a directed multigraph in NetworkX and persists to JSON.
Supports neighbour-based retrieval: given query entities, expand to
neighbouring nodes and return the associated chunk metadata.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import networkx as nx

from ...domain.ports import GraphStore
from ...domain.value_objects import ChunkMetadata, RetrievalResult, Triple

log = logging.getLogger(__name__)


class NetworkXGraphStore(GraphStore):
    """Concrete :class:`GraphStore` backed by a NetworkX directed multigraph."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._graph: nx.MultiDiGraph
        self._triple_sources: dict[int, Triple] = {}
        self._entity_to_chunks: dict[str, list[ChunkMetadata]] = {}
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

    def _extract_entity_mentions(self, question: str) -> list[str]:
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

    def _load_or_create(self) -> None:
        graph_file = self._path / "graph.json"
        chunks_file = self._path / "entity_chunks.json"
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
