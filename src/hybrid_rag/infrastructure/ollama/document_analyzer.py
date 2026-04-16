"""Ollama-based document analyzer — implements :class:`DocumentAnalyzer`.

Uses an LLM served by Ollama to analyse a document's structure and suggest
which entity-relationship patterns a knowledge graph should capture.  The
resulting :class:`DocumentAnalysis` is passed to the :class:`TripleExtractor`
as guidance so it can produce more focused extractions.
"""

from __future__ import annotations

import json
import logging
import re
import time

from ...domain.ports import DocumentAnalyzer
from ...domain.value_objects import DocumentAnalysis
from .client import OllamaClient

log = logging.getLogger(__name__)

_MAX_ANALYSIS_CHARS = 3000

_ANALYSIS_PROMPT = """\
Analyse the following document text and answer two questions:

1. **What is this document?** — Give a short type label (e.g. "clinical trial report", \
"SEC 10-K filing", "research paper", "product manual") and a one-sentence description.

2. **What knowledge-graph triple patterns would be most useful?** — List the \
entity-relationship patterns that a knowledge graph should capture from this \
kind of document. Each pattern should be in the form \
"subject_type → relationship → object_type".
For example: "company → founded_by → person", "drug → tested_for → indication", \
"product → has_price → amount".

Return ONLY a JSON object with exactly these keys:
{{
  "document_type": "short type label",
  "description": "one-sentence description",
  "suggested_triple_patterns": ["pattern1", "pattern2", ...]
}}

Do NOT include any other text, explanation, or markdown formatting.

Document text:
{text}"""


class OllamaDocumentAnalyzer(DocumentAnalyzer):
    """Concrete :class:`DocumentAnalyzer` backed by the Ollama HTTP API."""

    def __init__(self, client: OllamaClient, model: str) -> None:
        self._client = client
        self._model = model

    def analyze(self, text: str) -> DocumentAnalysis:
        sample = text[:_MAX_ANALYSIS_CHARS]
        log.info(
            "Analyzing document model=%s text_len=%d sample_len=%d",
            self._model,
            len(text),
            len(sample),
        )
        t0 = time.perf_counter()
        prompt = _ANALYSIS_PROMPT.format(text=sample)
        resp = self._client.post(
            "/api/generate",
            body={"model": self._model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        raw = resp.json()["response"]
        elapsed = time.perf_counter() - t0
        analysis = self._parse(raw)
        log.info(
            "Document analysis done model=%s doc_type=%s patterns=%d response_len=%d (%.2fs)",
            self._model,
            analysis.doc_type,
            len(analysis.suggested_triple_patterns),
            len(raw),
            elapsed,
        )
        for pattern in analysis.suggested_triple_patterns:
            log.debug("  suggested pattern: %s", pattern)
        return analysis

    @staticmethod
    def _parse(raw: str) -> DocumentAnalysis:
        cleaned = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            log.warning("No JSON object found in analyser output: %s", raw[:300])
            return DocumentAnalysis()

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            log.warning("Failed to parse JSON from analyser output: %s", raw[:300])
            return DocumentAnalysis()

        doc_type = str(data.get("document_type", "")).strip()
        description = str(data.get("description", "")).strip()

        patterns_raw = data.get("suggested_triple_patterns", [])
        if not isinstance(patterns_raw, list):
            patterns_raw = []
        patterns = [str(p).strip() for p in patterns_raw if str(p).strip()]

        return DocumentAnalysis(
            doc_type=doc_type,
            doc_description=description,
            suggested_triple_patterns=patterns,
        )
