"""Domain ports — abstract base classes that define contracts for infrastructure.

Infrastructure implementations (adapters) must subclass these ABCs and provide
concrete behaviour.  The domain and application layers depend only on these
abstractions, never on concrete infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .entities import Document
from .value_objects import ChunkMetadata, EmbeddingVector, RetrievalResult, Triple


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
    def add(
        self, vectors: list[EmbeddingVector], metadata: list[ChunkMetadata]
    ) -> None:
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
    """Port for a knowledge graph store.

    Stores entity-relation triples extracted from documents and supports
    neighbour-based graph retrieval for hybrid search.
    """

    @abstractmethod
    def add_triples(self, triples: list[Triple]) -> None:
        """Add triples to the graph."""
        ...

    @abstractmethod
    def query(self, question: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve chunks connected to entities mentioned in *question*."""
        ...

    @abstractmethod
    def node_count(self) -> int:
        """Return the number of unique nodes in the graph."""
        ...

    @abstractmethod
    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        ...

    @abstractmethod
    def all_triples(self) -> list[Triple]:
        """Return all triples in the graph."""
        ...

    def extract_entity_mentions(self, question: str) -> list[str]:
        """Return entity nodes in the graph that match the question.

        Default implementation returns an empty list; adapters may override.
        """
        return []


class TripleExtractor(ABC):
    """Port for extracting knowledge-graph triples from text."""

    @abstractmethod
    def extract(self, text: str, source: str) -> list[Triple]:
        """Extract (subject, predicate, object) triples from *text*."""
        ...
