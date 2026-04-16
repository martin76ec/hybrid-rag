"""Ollama-based triple extractor — implements :class:`TripleExtractor`.

Uses an LLM served by Ollama to extract (subject, predicate, object) triples
from text.  Returns parsed triples as :class:`Triple` value objects.
"""

from __future__ import annotations

import json
import logging
import re
import time

from ...domain.ports import TripleExtractor
from ...domain.value_objects import DocumentAnalysis, Triple
from .client import OllamaClient

log = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Extract knowledge-graph triples from the following text.
Return ONLY a JSON array of objects with keys "subject", "predicate", "object".
Each value must be a non-null string.
Do NOT include any other text, explanation, or markdown formatting.

Text:
{text}"""

_GUIDED_EXTRACTION_PROMPT = """\
Extract knowledge-graph triples from the following text.

Document context: This is a {doc_type}. {doc_description}
Focus on extracting these relationship patterns:
{patterns}

Return ONLY a JSON array of objects with keys "subject", "predicate", "object".
Each value must be a non-null string.
Do NOT include any other text, explanation, or markdown formatting.

Text:
{text}"""


class OllamaTripleExtractor(TripleExtractor):
    """Concrete :class:`TripleExtractor` backed by the Ollama HTTP API."""

    def __init__(self, client: OllamaClient, model: str) -> None:
        self._client = client
        self._model = model
        self._guidance: DocumentAnalysis | None = None

    def set_guidance(self, guidance: DocumentAnalysis | None) -> None:
        self._guidance = guidance
        if guidance:
            log.info(
                "Extractor guidance set: doc_type=%s patterns=%d",
                guidance.doc_type,
                len(guidance.suggested_triple_patterns),
            )

    def extract(self, text: str, source: str = "") -> list[Triple]:
        log.info(
            "Extract triples model=%s text_len=%d source=%s guided=%s",
            self._model,
            len(text),
            source,
            bool(self._guidance),
        )
        t0 = time.perf_counter()
        prompt = self._build_prompt(text)
        resp = self._client.post(
            "/api/generate",
            body={"model": self._model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        raw = resp.json()["response"]
        elapsed = time.perf_counter() - t0
        triples = self._parse(raw, source)
        log.info(
            "Extract triples done model=%s triples=%d response_len=%d (%.2fs)",
            self._model,
            len(triples),
            len(raw),
            elapsed,
        )
        return triples

    @staticmethod
    def _parse(raw: str, source: str) -> list[Triple]:
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            try:
                items = json.loads(json_match.group())
                if isinstance(items, list):
                    return OllamaTripleExtractor._from_dicts(items, source)
            except json.JSONDecodeError:
                log.warning("JSON array found but failed to parse: %s", raw[:200])

        triples = OllamaTripleExtractor._parse_line_format(raw, source)
        if triples:
            return triples

        log.warning("No valid triples found in LLM output: %s", raw[:200])
        return []

    def _build_prompt(self, text: str) -> str:
        if self._guidance and (
            self._guidance.doc_type or self._guidance.suggested_triple_patterns
        ):
            patterns = (
                "\n".join(f"- {p}" for p in self._guidance.suggested_triple_patterns)
                if self._guidance.suggested_triple_patterns
                else "(use general judgement)"
            )
            return _GUIDED_EXTRACTION_PROMPT.format(
                doc_type=self._guidance.doc_type or "document",
                doc_description=self._guidance.doc_description or "",
                patterns=patterns,
                text=text,
            )
        return _EXTRACTION_PROMPT.format(text=text)

    @staticmethod
    def _from_dicts(items: list, source: str) -> list[Triple]:
        triples: list[Triple] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            s = str(item.get("subject") or "").strip()
            p = str(item.get("predicate") or "").strip()
            o = str(item.get("object") or "").strip()
            if (
                s
                and p
                and o
                and s.lower() != "null"
                and p.lower() != "null"
                and o.lower() != "null"
            ):
                triples.append(Triple(subject=s, predicate=p, obj=o, source=source))
        return triples

    @staticmethod
    def _parse_line_format(raw: str, source: str) -> list[Triple]:
        pattern = re.compile(
            r'\{"subject"\s*:\s*"([^"]+)"\s*,\s*"predicate"\s*:\s*"([^"]+)"\s*,\s*"object"\s*:\s*"([^"]+)"\s*\}'
        )
        triples: list[Triple] = []
        for m in pattern.finditer(raw):
            triples.append(
                Triple(
                    subject=m.group(1).strip(),
                    predicate=m.group(2).strip(),
                    obj=m.group(3).strip(),
                    source=source,
                )
            )
        return triples
