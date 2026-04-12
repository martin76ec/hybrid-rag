"""Domain entities.

Entities are objects defined by their identity rather than their attributes.
In this project the core entities are *Document* (a source file that has been
read) and *Chunk* (a fixed-size slice of a document's text).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """A document that has been read from an external source.

    Attributes:
        source: File path or URI that identifies where the document came from.
        text:   Raw text extracted from the document.
    """

    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A fixed-size slice of a :class:`Document`.

    Attributes:
        text:        The text content of the chunk.
        source:      The source identifier inherited from the parent document.
        chunk_index: Position of this chunk within the original document.
    """

    text: str
    source: str
    chunk_index: int