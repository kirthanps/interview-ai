"""Extract selectable text from an uploaded resume PDF."""

import logging
from pathlib import Path
from typing import List

import PyPDF2

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF file.

    Args:
        file_path: Path to PDF file.

    Returns:
        Concatenated text from all pages.

    Raises:
        FileNotFoundError: If PDF path does not exist.
        ValueError: If no extractable text is found.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        text_parts: List[str] = []
        with path.open("rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            logger.info("Extracting text from PDF with %s pages", num_pages)

            for page_number, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
                logger.debug("Processed page %s/%s", page_number, num_pages)

        text = "\n".join(text_parts).strip()
        if not text:
            raise ValueError(
                "No text found in PDF. The file may be image-based and require OCR."
            )

        logger.info("Successfully extracted %s characters from PDF", len(text))
        return text

    except Exception as exc:
        logger.error("Error extracting text from PDF %s: %s", file_path, exc)
        raise
