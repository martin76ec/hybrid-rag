"""Ollama embedding provider — implements :class:`EmbeddingProvider`.

Uses the Ollama HTTP API (``/api/embeddings``) to produce dense vectors from
text.
"""

from __future__ import annotations

import logging
import time

from ...domain.ports import EmbeddingProvider
from ...domain.value_objects import EmbeddingVector
from .client import OllamaClient

log = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Concrete :class:`EmbeddingProvider` backed by the Ollama HTTP API."""

    def __init__(self, client: OllamaClient, model: str) -> None:
        self._client = client
        self._model = model

    def embed(self, text: str) -> EmbeddingVector:
        log.debug("Embed text_len=%d model=%s", len(text), self._model)
        t0 = time.perf_counter()
        resp = self._client.post(
            "/api/embeddings", body={"model": self._model, "prompt": text}, timeout=30
        )
        vec = EmbeddingVector(values=resp.json()["embedding"])
        elapsed = time.perf_counter() - t0
        log.info(
            "Embed done model=%s dim=%d (%.2fs)", self._model, len(vec.values), elapsed
        )
        return vec

    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        log.info("Embed batch size=%d model=%s", len(texts), self._model)
        results: list[EmbeddingVector] = []
        for i, text in enumerate(texts, 1):
            log.debug("Embed batch %d/%d", i, len(texts))
            results.append(self.embed(text))
        log.info("Embed batch done model=%s count=%d", self._model, len(results))
        return results
