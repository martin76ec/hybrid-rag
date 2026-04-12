"""Configuration utilities.

Loads environment variables from a ``.env`` file (if present) and provides a
simple ``Config`` dataclass that other modules can import.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load a ``.env`` file from the project root if it exists.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = PROJECT_ROOT / ".env"
if DOTENV_PATH.is_file():
    load_dotenv(dotenv_path=DOTENV_PATH)


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables.

    Expected variables:
    - ``OLLAMA_HOST`` – URL of the Ollama server (default ``http://localhost:11434``)
    - ``EMBEDDING_MODEL`` – Ollama embedding model name (default ``nomic-embed-text``)
    - ``LLM_MODEL`` – Ollama LLM model name (default ``llama3.2``)
    - ``FAISS_INDEX_PATH`` – path to persist the vector store (default
      ``<project_root>/faiss_index``)
    - ``GRAPH_STORE_PATH`` – path to persist the knowledge graph (default
      ``<project_root>/graph_store``)
    """

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2")
    faiss_index_path: str = os.getenv(
        "FAISS_INDEX_PATH", str(PROJECT_ROOT / "faiss_index")
    )
    graph_store_path: str = os.getenv(
        "GRAPH_STORE_PATH", str(PROJECT_ROOT / "graph_store")
    )
