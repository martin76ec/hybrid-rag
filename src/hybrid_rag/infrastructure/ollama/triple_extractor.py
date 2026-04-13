"""Ollama-based triple extractor — implements :class:`TripleExtractor`.

Uses an LLM served by Ollama to extract (subject, predicate, object) triples
from text.  Returns parsed triples as :class:`Triple` value objects.
"""

from __future__ import annotations

import json
import logging
import re

import requests

from ...domain.ports import TripleExtractor
from ...domain.value_objects import Triple

log = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Extract knowledge-graph triples from the following text.
Return ONLY a JSON array of objects with keys "subject", "predicate", "object".
Each value must be a non-null string.
Do NOT include any other text, explanation, or markdown formatting.

Text:
{text}"""


class OllamaTripleExtractor(TripleExtractor):
    """Concrete :class:`TripleExtractor` backed by the Ollama HTTP API."""

    def __init__(self, host: str, model: str) -> None:
        self._url = f"{host.rstrip('/')}/api/generate"
        self._model = model

    def extract(self, text: str, source: str = "") -> list[Triple]:
        prompt = _EXTRACTION_PROMPT.format(text=text)
        resp = requests.post(
            self._url,
            json={"model": self._model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["response"]
        return self._parse(raw, source)

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
