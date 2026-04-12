"""Ollama embedding provider — implements :class:`EmbeddingProvider`.

Uses the Ollama HTTP API (``/api/embeddings``) to produce dense vectors from
text.
"""

from __future__ import annotations

import logging
from typing import List

import requests

from ...domain.ports import EmbeddingProvider
from ...domain.value_objects import EmbeddingVector

log = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Concrete :class:`EmbeddingProvider` backed by the Ollama HTTP API."""

    def __init__(self, host: str, model: str) -> None:
        self._url = f"{host.rstrip('/')}/api/embeddings"
        self._model = model

    def embed(self, text: str) -> EmbeddingVector:
        """Embed a single piece of text via Ollama."""
        resp = requests.post(
            self._url,
            json={"model": self._model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return EmbeddingVector(values=resp.json()["embedding"])

    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed multiple texts by calling :meth:`embed` sequentially.

        Ollama does not expose a native batch endpoint, so each text is
        embedded individually.
        """
        return [self.embed(text) for text in texts]