"""Render knowledge graph HTML using vis-network.js (same engine as Neo4j Browser).

Generates a standalone HTML file with graph data embedded as JSON. No
external server required — vis-network.js is loaded from CDN.
"""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent


def render_graph_html(
    nodes: list[dict],
    edges: list[dict],
    output_path: str,
    highlighted_nodes: list[str] | None = None,
) -> str:
    """Render a vis-network HTML file with embedded graph data.

    Args:
        nodes:            List of node dicts with at least ``id``, ``label``,
                          ``color``, ``_source`` keys.
        edges:            List of edge dicts with ``from``, ``to``, ``label``.
        output_path:      Where to write the HTML file.
        highlighted_nodes: Node IDs to highlight (yellow, larger).

    Returns:
        The output_path string.
    """
    template = (TEMPLATE_DIR / "graph.html").read_text(encoding="utf-8")

    graph_data = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    highlighted = json.dumps(highlighted_nodes or [])

    html = template.replace("__GRAPH_DATA__", graph_data).replace(
        "__HIGHLIGHTED_NODES__", highlighted
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
