"""PyPDF document reader — implements :class:`DocumentReader`.

Extracts raw text from PDF files using **pypdf**.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from ...domain.entities import Document
from ...domain.ports import DocumentReader


class PyPDFDocumentReader(DocumentReader):
    """Concrete :class:`DocumentReader` that reads PDF files."""

    def read(self, source: str) -> Document:
        """Read a PDF file at *source* and return its extracted text."""
        reader = PdfReader(source)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return Document(source=source, text=text)