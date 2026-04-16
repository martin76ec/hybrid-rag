"""Ollama language model — implements :class:`LanguageModel`.

Uses the Ollama HTTP API (``/api/generate``) to produce text completions.
"""

from __future__ import annotations

import logging
import time

from ...domain.ports import LanguageModel
from .client import OllamaClient

log = logging.getLogger(__name__)


class OllamaLanguageModel(LanguageModel):
    """Concrete :class:`LanguageModel` backed by the Ollama HTTP API."""

    def __init__(self, client: OllamaClient, model: str) -> None:
        self._client = client
        self._model = model

    def generate(self, prompt: str) -> str:
        log.info("LLM generate model=%s prompt_len=%d", self._model, len(prompt))
        t0 = time.perf_counter()
        resp = self._client.post(
            "/api/generate",
            body={"model": self._model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        elapsed = time.perf_counter() - t0
        text = resp.json()["response"]
        log.info(
            "LLM generate done model=%s response_len=%d (%.2fs)",
            self._model,
            len(text),
            elapsed,
        )
        return text
