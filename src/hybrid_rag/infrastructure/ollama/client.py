"""Shared Ollama HTTP client with retry logic, concurrency limiting, and LLM caching.

All Ollama adapters should go through this client instead of calling
``requests`` directly.  It provides:

- **Exponential-backoff retry** on transient errors (502, 503, 429, timeouts)
- **Concurrency semaphore** so at most *max_concurrency* requests are in-flight
  at once, preventing Ollama from being overwhelmed
- **LLM response cache** — ``/api/generate`` responses are cached to disk so
  repeated identical calls return instantly (embedding endpoints are not cached)
"""

from __future__ import annotations

import json
import logging
import time
from threading import Semaphore

import requests

from .cache import LLMCache

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 502, 503, 504}
_CACHEABLE_ENDPOINTS = {"/api/generate"}


class OllamaClient:
    """Thread-safe HTTP client for Ollama with retry, concurrency, and caching.

    Args:
        host:            Ollama base URL (e.g. ``http://localhost:11434``).
        max_retries:     Max retry attempts per request (0 = no retry).
        max_concurrency: Max simultaneous in-flight requests.
        base_backoff:    Seconds to wait before first retry (doubles each attempt).
        cache_dir:       Directory for LLM response cache. ``None`` disables caching.
    """

    def __init__(
        self,
        host: str,
        max_retries: int = 3,
        max_concurrency: int = 3,
        base_backoff: float = 2.0,
        cache_dir: str | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._semaphore = Semaphore(max_concurrency)
        self._cache = LLMCache(cache_dir)

    def post(self, endpoint: str, body: dict, timeout: int = 120) -> requests.Response:
        """Send a POST request to an Ollama endpoint.

        For ``/api/generate``, responses are cached on disk keyed by
        ``(model, prompt)``.  A cache hit returns a synthetic
        ``requests.Response`` with status 200 — no HTTP call is made.

        Args:
            endpoint: Path under the Ollama host (e.g. ``/api/generate``).
            body:    JSON body for the request.
            timeout: Request timeout in seconds.

        Returns:
            The successful ``requests.Response``.

        Raises:
            requests.HTTPError: If the request fails after all retries.
        """
        if endpoint in _CACHEABLE_ENDPOINTS:
            cached = self._cache.get(body.get("model", ""), body.get("prompt", ""))
            if cached is not None:
                resp = requests.Response()
                resp.status_code = 200
                resp._content = f'{{"response":{json.dumps(cached)}}}'.encode()
                log.info(
                    "LLM cache hit model=%s prompt_len=%d endpoint=%s",
                    body.get("model", ""),
                    len(body.get("prompt", "")),
                    endpoint,
                )
                return resp

        url = f"{self._host}{endpoint}"
        last_exc: Exception | None = None

        log.info(
            "POST %s model=%s prompt_len=%d timeout=%ds",
            endpoint,
            body.get("model", ""),
            len(body.get("prompt", "")),
            timeout,
        )

        with self._semaphore:
            for attempt in range(self._max_retries + 1):
                t0 = time.perf_counter()
                try:
                    resp = requests.post(url, json=body, timeout=timeout)
                    elapsed = time.perf_counter() - t0
                    if resp.status_code not in _RETRYABLE_STATUS:
                        resp.raise_for_status()
                        log.info(
                            "POST %s → %d OK (%.2fs)",
                            endpoint,
                            resp.status_code,
                            elapsed,
                        )
                        if endpoint in _CACHEABLE_ENDPOINTS:
                            self._cache.put(
                                body.get("model", ""),
                                body.get("prompt", ""),
                                resp.json()["response"],
                            )
                        return resp

                    last_exc = requests.HTTPError(
                        f"{resp.status_code} {resp.reason} for url: {url}"
                    )
                    log.warning(
                        "POST %s → %d (attempt %d/%d, %.2fs)",
                        endpoint,
                        resp.status_code,
                        attempt + 1,
                        self._max_retries + 1,
                        elapsed,
                    )
                except requests.Timeout as exc:
                    last_exc = exc
                    elapsed = time.perf_counter() - t0
                    log.warning(
                        "POST %s timed out (attempt %d/%d, %.2fs)",
                        endpoint,
                        attempt + 1,
                        self._max_retries + 1,
                        elapsed,
                    )
                except requests.ConnectionError as exc:
                    last_exc = exc
                    elapsed = time.perf_counter() - t0
                    log.warning(
                        "POST %s connection error (attempt %d/%d, %.2fs)",
                        endpoint,
                        attempt + 1,
                        self._max_retries + 1,
                        elapsed,
                    )
                except requests.HTTPError as exc:
                    if "404" in str(exc) or "400" in str(exc):
                        raise
                    last_exc = exc

                if attempt < self._max_retries:
                    delay = self._base_backoff * (2**attempt)
                    log.info("Retrying in %.1fs…", delay)
                    time.sleep(delay)

        raise last_exc  # type: ignore[misc]
