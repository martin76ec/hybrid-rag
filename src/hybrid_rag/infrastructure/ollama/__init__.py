"""Ollama infrastructure adapters — embedding provider, language model, and triple extractor."""

from .embeddings import OllamaEmbeddingProvider
from .llm import OllamaLanguageModel
from .triple_extractor import OllamaTripleExtractor

__all__ = ["OllamaEmbeddingProvider", "OllamaLanguageModel", "OllamaTripleExtractor"]
