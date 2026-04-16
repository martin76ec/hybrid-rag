"""Structured logging configuration for Hybrid-RAG.

Configures the root ``hybrid_rag`` logger with a Rich-based console handler
that produces coloured, formatted output.  Call :func:`setup_logging` once at
application startup (from the CLI or web entry-points).  The effective log
level can be controlled via:

* The ``LOG_LEVEL`` environment variable (default ``INFO``).
* CLI flags ``--verbose`` (INFO) and ``--debug`` (DEBUG) which override the
  environment variable.
"""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

_LOGGER_NAME = "hybrid_rag"


def setup_logging(level: str | None = None) -> None:
    """Configure the ``hybrid_rag`` logger with a Rich console handler.

    Args:
        level: Log level string (e.g. ``"DEBUG"``, ``"INFO"``).  Falls back
               to the ``LOG_LEVEL`` env var, then to ``"INFO"``.
    """
    effective = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, effective, logging.INFO)

    handler = RichHandler(
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(numeric)

    if not logger.handlers:
        logger.addHandler(handler)
    else:
        logger.handlers.clear()
        logger.addHandler(handler)

    logger.propagate = False
