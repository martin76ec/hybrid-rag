"""Ollama-based triple refiner — implements :class:`TripleRefiner`.

Sends raw triples to the LLM in batches and asks it to:
1. Map duplicate entities to a canonical form
2. Shorten overly long predicates
3. Remove trivial/useless triples
4. Suggest logically implied edges that are missing

Batches are sized to stay within LLM context limits (~35 triples per call).
Results are merged across batches with first-batch-wins conflict resolution.
"""

from __future__ import annotations

import json
import logging
import re
import time

from ...domain.ports import TripleRefiner
from ...domain.value_objects import Triple
from .client import OllamaClient

log = logging.getLogger(__name__)

_BATCH_SIZE = 35

_REFINE_PROMPT = """\
You are a knowledge-graph engineer. Given the following raw triples extracted from documents, perform these tasks:

1. **canonical_entities**: Map duplicate or variant entity names to a single canonical form.
   For example, "NovaMind" and "NovaMind Technologies" should both map to "NovaMind Technologies".
   Only include entities that actually need remapping (where the key and value are different).
   Values must be strings, not objects.

2. **shortened_predicates**: Shorten overly long predicates to 1-3 word forms.
   For example, "was chaired by the board of directors" → "chairs".
   Only include predicates that need shortening.

3. **removed_indices**: List the 0-based indices of triples that are trivial, useless, or contain no real information.
   Remove triples where the predicate is just "is", "was", "has" with no meaningful object, or where the subject or object is null/empty/a pronoun.

4. **added_triples**: Suggest new triples that are logically implied by the text but were missed.
   For example, if "NovaMind Technologies founded by Elena Vasquez" and "NovaMind Technologies develops Cortex-7" appear, you might add (NovaMind Technologies, develops, Cortex-7) if it was missed.
   Only add triples that are clearly supported by the existing ones. Use canonical entity names.
   Be moderate — only add edges that are explicitly stated or very strongly implied, not common-sense inferences.

Raw triples:
{triples}

IMPORTANT: Return ONLY valid JSON. No markdown, no code fences, no explanation. Just the JSON object:
{{
  "canonical_entities": {{}},
  "shortened_predicates": [],
  "removed_indices": [],
  "added_triples": []
}}"""


class OllamaTripleRefiner(TripleRefiner):
    """Concrete :class:`TripleRefiner` backed by the Ollama HTTP API.

    Processes triples in batches of 35 to stay within LLM context limits.
    """

    def __init__(self, client: OllamaClient, model: str) -> None:
        self._client = client
        self._model = model

    def refine(self, raw_triples: list[Triple]) -> dict:
        if not raw_triples:
            return self._empty_result()

        batches = [
            raw_triples[i : i + _BATCH_SIZE]
            for i in range(0, len(raw_triples), _BATCH_SIZE)
        ]
        total_batches = len(batches)
        log.info("Refining %d triples in %d batches", len(raw_triples), total_batches)

        merged_mapping: dict[str, str] = {}
        merged_shortened: list[tuple[str, str]] = []
        seen_shortened: set[str] = set()
        all_removed_indices: list[int] = []
        merged_added: list[tuple[str, str, str]] = []
        seen_added: set[tuple[str, str, str]] = set()

        for batch_idx, batch in enumerate(batches, start=1):
            batch_start = (batch_idx - 1) * _BATCH_SIZE
            result = self._refine_batch(batch, batch_start, batch_idx, total_batches)

            for raw, canonical in result.get("canonical_mapping", {}).items():
                if raw != canonical and raw not in merged_mapping:
                    merged_mapping[raw] = canonical

            for before, after in result.get("shortened_predicates", []):
                if before not in seen_shortened:
                    merged_shortened.append((before, after))
                    seen_shortened.add(before)

            all_removed_indices.extend(result.get("removed_indices", []))

            for s, p, o in result.get("added_triples", []):
                key = (s, p, o)
                if key not in seen_added:
                    merged_added.append((s, p, o))
                    seen_added.add(key)

        removed_triples = []
        for idx in sorted(set(all_removed_indices)):
            if 0 <= idx < len(raw_triples):
                t = raw_triples[idx]
                removed_triples.append((t.subject, t.predicate, t.obj))

        log.info(
            "Refiner merged totals: %d canonical, %d shortened, %d removed, %d added",
            len(merged_mapping),
            len(merged_shortened),
            len(removed_triples),
            len(merged_added),
        )
        if merged_mapping:
            log.info("Canonical mappings: %s", merged_mapping)

        return {
            "canonical_mapping": merged_mapping,
            "shortened_predicates": merged_shortened,
            "removed_triples": removed_triples,
            "added_triples": merged_added,
        }

    def _refine_batch(
        self,
        batch: list[Triple],
        batch_start: int,
        batch_idx: int,
        total_batches: int,
    ) -> dict:
        triples_text = "\n".join(
            f"{batch_start + i}. ({t.subject}, {t.predicate}, {t.obj})"
            for i, t in enumerate(batch)
        )
        prompt = _REFINE_PROMPT.format(triples=triples_text)

        try:
            t0 = time.perf_counter()
            resp = self._client.post(
                "/api/generate",
                body={"model": self._model, "prompt": prompt, "stream": False},
                timeout=300,
            )
            raw = resp.json()["response"]
            elapsed = time.perf_counter() - t0
            log.info(
                "Refiner batch %d/%d LLM call done model=%s response_len=%d (%.2fs)",
                batch_idx,
                total_batches,
                self._model,
                len(raw),
                elapsed,
            )
        except Exception as exc:
            log.warning(
                "Refiner batch %d/%d call failed: %s", batch_idx, total_batches, exc
            )
            return self._empty_result()

        log.debug(
            "Refiner batch %d/%d raw response (first 500 chars): %s",
            batch_idx,
            total_batches,
            raw[:500],
        )
        parsed = self._parse(raw, batch, batch_start)

        log.info(
            "Refiner batch %d/%d: %d canonical, %d shortened, %d removed, %d added",
            batch_idx,
            total_batches,
            len(parsed.get("canonical_mapping", {})),
            len(parsed.get("shortened_predicates", [])),
            len(parsed.get("removed_indices", [])),
            len(parsed.get("added_triples", [])),
        )

        return parsed

    @staticmethod
    def _parse(raw: str, batch: list[Triple], batch_start: int) -> dict:
        # Strip markdown code fences if present
        cleaned = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            log.warning(
                "No JSON object found in refiner output (first 300 chars): %s",
                raw[:300],
            )
            return {
                "canonical_mapping": {},
                "shortened_predicates": [],
                "removed_indices": [],
                "added_triples": [],
            }

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            log.warning(
                "Failed to parse JSON from refiner output (first 300 chars): %s",
                raw[:300],
            )
            return {
                "canonical_mapping": {},
                "shortened_predicates": [],
                "removed_indices": [],
                "added_triples": [],
            }

        raw_mapping = data.get("canonical_entities", {})
        canonical_mapping: dict[str, str] = {}
        if isinstance(raw_mapping, dict):
            for key, value in raw_mapping.items():
                if not isinstance(key, str) or not key.strip():
                    continue
                if isinstance(value, str):
                    if key.strip() != value.strip():
                        canonical_mapping[key.strip()] = value.strip()
                elif isinstance(value, dict):
                    for v in value.values():
                        if isinstance(v, str) and v.strip():
                            if key.strip() != v.strip():
                                canonical_mapping[key.strip()] = v.strip()
                            break

        shortened = data.get("shortened_predicates", [])
        if not isinstance(shortened, list):
            shortened = []
        shortened_predicates = []
        for s in shortened:
            if isinstance(s, dict) and s.get("before") and s.get("after"):
                shortened_predicates.append((s["before"], s["after"]))
            elif isinstance(s, str):
                pass  # LLM returned just the before string — skip

        removed_indices = data.get("removed_indices", [])
        if not isinstance(removed_indices, list):
            removed_indices = []
        absolute_indices = []
        for idx in removed_indices:
            if isinstance(idx, int) and 0 <= idx < len(batch):
                absolute_indices.append(batch_start + idx)

        added_raw = data.get("added_triples", [])
        if not isinstance(added_raw, list):
            added_raw = []
        added_triples = []
        for item in added_raw:
            if not isinstance(item, dict):
                continue
            # Handle malformed added_triples where keys are entity names instead of subject/predicate/object
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
                added_triples.append((s, p, o))

        return {
            "canonical_mapping": canonical_mapping,
            "shortened_predicates": shortened_predicates,
            "removed_indices": absolute_indices,
            "added_triples": added_triples,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "canonical_mapping": {},
            "shortened_predicates": [],
            "removed_indices": [],
            "added_triples": [],
        }
