"""Ollama language model — implements :class:`LanguageModel`.

Uses the Ollama HTTP API (``/api/generate``) to produce text completions.
"""

from __future__ import annotations

import logging

import requests

from ...domain.ports import LanguageModel

log = logging.getLogger(__name__)


class OllamaLanguageModel(LanguageModel):
    """Concrete :class:`LanguageModel` backed by the Ollama HTTP API."""

    def __init__(self, host: str, model: str) -> None:
        self._url = f"{host.rstrip('/')}/api/generate"
        self._model = model

    def generate(self, prompt: str) -> str:
        """Generate a completion for *prompt* via Ollama."""
        resp = requests.post(
            self._url,
            json={"model": self._model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"]