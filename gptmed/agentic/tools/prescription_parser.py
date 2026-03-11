"""
Prescription Parser - File → Raw Text Extraction

Handles three input types:
    - PDF  → text via PyPDF2 (no extra system deps)
    - Image (JPEG, PNG, TIFF, BMP, WEBP) → text via pytesseract + Pillow
    - Plain text / unknown → decoded as UTF-8

The output of this module is always a plain-text string, which is then
passed directly into PrescriptionAnalyzerAgent (which already handles
raw text input).

Dependencies
────────────
PDF:    PyPDF2  (already in requirements.txt)
Image:  Pillow  (already in requirements.txt)
        pytesseract  → pip install pytesseract
        Tesseract binary → sudo apt install tesseract-ocr   (Linux)
                             brew install tesseract           (macOS)
"""

from __future__ import annotations

import io
from typing import Optional


# ── MIME / extension helpers ──────────────────────────────────────────────────

_PDF_TYPES = {"application/pdf"}
_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/tiff", "image/bmp", "image/webp",
}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def _is_pdf(mime_type: str, filename: str) -> bool:
    return mime_type in _PDF_TYPES or filename.lower().endswith(".pdf")


def _is_image(mime_type: str, filename: str) -> bool:
    if mime_type in _IMAGE_TYPES:
        return True
    import os
    return os.path.splitext(filename.lower())[1] in _IMAGE_EXTS


# ── Core class ────────────────────────────────────────────────────────────────

class PrescriptionParser:
    """
    Converts uploaded prescription files into raw text strings.

    Usage::

        text = PrescriptionParser.parse(file_bytes, "application/pdf", "rx.pdf")
        text = PrescriptionParser.parse(file_bytes, "image/png",       "rx.png")
    """

    @staticmethod
    def parse(file_bytes: bytes, mime_type: str, filename: str) -> str:
        """
        Extract text from a prescription file.

        Args:
            file_bytes: Raw file content.
            mime_type:  MIME type reported by the HTTP client (e.g. "application/pdf").
            filename:   Original filename (used as fallback type detection).

        Returns:
            Extracted plain text string.

        Raises:
            ImportError:  If pytesseract / tesseract binary is missing for images.
            ValueError:   If the file is empty or text extraction yields nothing.
        """
        mime_type = (mime_type or "").lower()
        filename = filename or ""

        if _is_pdf(mime_type, filename):
            text = PrescriptionParser._parse_pdf(file_bytes)
        elif _is_image(mime_type, filename):
            text = PrescriptionParser._parse_image(file_bytes)
        else:
            # treat as plain text
            text = file_bytes.decode("utf-8", errors="replace")

        text = text.strip()
        if not text:
            raise ValueError(
                "No text could be extracted from the uploaded file. "
                "Please ensure the file is not empty or image-only without readable text."
            )
        return text

    # ── Internal parsers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_pdf(file_bytes: bytes) -> str:
        """Extract text from a PDF using PyPDF2 (already a project dependency)."""
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PyPDF2 is required for PDF parsing. "
                "Install it with: pip install PyPDF2"
            ) from exc

        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

        return "\n".join(pages_text)

    @staticmethod
    def _parse_image(file_bytes: bytes) -> str:
        """
        Extract text from a prescription image using pytesseract OCR.

        Requirements:
            pip install pytesseract Pillow
            sudo apt install tesseract-ocr     # Linux
            brew install tesseract             # macOS
        """
        try:
            from PIL import Image
            import pytesseract
        except ImportError as exc:
            raise ImportError(
                "Image OCR requires pytesseract and Pillow.\n"
                "Install Python packages:\n"
                "    pip install pytesseract Pillow\n"
                "Install Tesseract binary:\n"
                "    Linux:  sudo apt install tesseract-ocr\n"
                "    macOS:  brew install tesseract\n"
                "    Windows: https://github.com/UB-Mannheim/tesseract/wiki"
            ) from exc

        image = Image.open(io.BytesIO(file_bytes))

        # Convert to RGB if needed (handles RGBA / palette modes)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Preprocess: convert to greyscale for better OCR accuracy
        grey = image.convert("L")

        text = pytesseract.image_to_string(grey, lang="eng")
        return text

    @staticmethod
    def supported_mime_types() -> list[str]:
        """Return the list of accepted MIME types."""
        return sorted(_PDF_TYPES | _IMAGE_TYPES)
