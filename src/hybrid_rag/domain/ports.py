"""Domain ports — abstract base classes that define contracts for infrastructure.

Infrastructure implementations (adapters) must subclass these ABCs and provide
concrete behaviour.  The domain and application layers depend only on these
abstractions, never on concrete infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .entities import Document
from .value_objects import ChunkMetadata, EmbeddingVector, RetrievalResult


class EmbeddingProvider(ABC):
    """Port for embedding text into dense vectors."""

    @abstractmethod
    def embed(self, text: str) -> EmbeddingVector:
        """Embed a single piece of text."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed multiple texts and return their vectors in order."""
        ...


class VectorStore(ABC):
    """Port for persisting and searching embedding vectors."""

    @abstractmethod
    def add(self, vectors: list[EmbeddingVector], metadata: list[ChunkMetadata]) -> None:
        """Add vectors and their associated metadata to the store."""
        ...

    @abstractmethod
    def search(self, query: EmbeddingVector, top_k: int) -> list[RetrievalResult]:
        """Return the *top_k* nearest neighbours for *query*."""
        ...


class LanguageModel(ABC):
    """Port for generating text from a prompt."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for *prompt*."""
        ...


class DocumentReader(ABC):
    """Port for reading raw text from an external document source."""

    @abstractmethod
    def read(self, source: str) -> Document:
        """Read a document from *source* (e.g. a file path or URI)."""
        ...


class GraphStore(ABC):
    """Placeholder port for a knowledge graph store.

    This port is included so the architecture is ready for the planned
    knowledge-graph feature.  No infrastructure implementation exists yet.
    """

    @abstractmethod
    def add_triples(self, triples: list[tuple[str, str, str]]) -> None:
        """Add (subject, predicate, object) triples to the graph."""
        ...

    @abstractmethod
    def query(self, question: str) -> list[dict]:
        """Query the graph and return structured results."""
        ...