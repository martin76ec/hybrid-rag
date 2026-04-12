"""Ollama infrastructure adapters — embedding provider and language model."""

from .embeddings import OllamaEmbeddingProvider
from .llm import OllamaLanguageModel

__all__ = ["OllamaEmbeddingProvider", "OllamaLanguageModel"]