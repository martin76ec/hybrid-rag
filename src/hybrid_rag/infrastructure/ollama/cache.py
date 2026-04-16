"""LLM response cache with JSONL disk persistence.

Caches ``/api/generate`` responses keyed on ``(model, prompt)`` so repeated
LLM calls with identical inputs return instantly.  Embedding endpoints are
NOT cached — they're already persisted in FAISS / entity vectors.

The cache is stored as a JSONL file where each line is a JSON object with
keys ``key`` (SHA-256 hex of ``model|prompt``), ``model``, ``prompt_hash``
(first 16 chars of the key), and ``response`` (the raw LLM output string).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class LLMCache:
    """Disk-backed cache for Ollama ``/api/generate`` responses.

    Args:
        cache_dir: Directory where the cache JSONL file lives.
                    Created on first write if it doesn't exist.
                    Pass ``None`` to disable caching entirely.
    """

    def __init__(self, cache_dir: str | None) -> None:
        self._entries: dict[str, str] = {}
        self._cache_file: Path | None = None
        self._enabled: bool = cache_dir is not None
        if self._enabled:
            self._cache_file = Path(cache_dir) / "llm_cache.jsonl"
            self._load()

    @staticmethod
    def _make_key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()

    def _load(self) -> None:
        if self._cache_file is None or not self._cache_file.is_file():
            return
        count = 0
        with self._cache_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._entries[obj["key"]] = obj["response"]
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        if count:
            log.info("LLM cache: loaded %d entries from %s", count, self._cache_file)

    def _append(self, key: str, response: str) -> None:
        if self._cache_file is None:
            return
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": response}) + "\n")

    def get(self, model: str, prompt: str) -> str | None:
        if not self._enabled:
            return None
        key = self._make_key(model, prompt)
        result = self._entries.get(key)
        if result is not None:
            log.debug("Cache hit model=%s", model)
        else:
            log.debug("Cache miss model=%s", model)
        return result

    def put(self, model: str, prompt: str, response: str) -> None:
        if not self._enabled:
            return
        key = self._make_key(model, prompt)
        if key in self._entries:
            return
        self._entries[key] = response
        self._append(key, response)
        log.debug("Cache store model=%s response_len=%d", model, len(response))

    @property
    def size(self) -> int:
        return len(self._entries)
