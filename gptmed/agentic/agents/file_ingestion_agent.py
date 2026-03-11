"""
File Ingestion Agent - Agent 0

Converts an uploaded file (PDF or image) into raw prescription text
that the downstream PrescriptionAnalyzerAgent can parse.

Input dict::

    {
        "file_bytes": <bytes>,    # raw file content
        "mime_type":  <str>,      # e.g. "application/pdf" or "image/png"
        "filename":   <str>,      # original filename for type fallback
    }

Output: dictionary containing extracted text and metadata.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict

# Support both package import and direct execution from agentic folder.
try:
    from ..tools.prescription_parser import PrescriptionParser
    from ..core.base_agent import BaseAgent
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tools.prescription_parser import PrescriptionParser
    from core.base_agent import BaseAgent


class FileIngestionAgent(BaseAgent):
    """
    Agent 0: File Ingestion

    Accepts a raw file upload payload and converts it to plain text
    via PrescriptionParser (PDF → PyPDF2, Image → pytesseract OCR).

    The extracted text is then forwarded to PrescriptionAnalyzerAgent,
    which already handles raw text strings as input.
    """

    SUPPORTED_MIME_TYPES = (
        "application/pdf",
        "image/jpeg", "image/jpg", "image/png",
        "image/tiff", "image/bmp", "image/webp",
    )

    def __init__(self) -> None:
        super().__init__(
            name="FileIngestionAgent",
            description="Converts uploaded PDF or image prescription files into raw text",
            enabled=True,
        )

    def validate_input(self, input_data: Any) -> bool:
        if not isinstance(input_data, dict):
            return False
        has_bytes = isinstance(input_data.get("file_bytes"), (bytes, bytearray))
        has_type = isinstance(input_data.get("mime_type"), str)
        has_name = isinstance(input_data.get("filename"), str)
        return has_bytes and has_type and has_name

    def validate_output(self, output: Any) -> bool:
        if not isinstance(output, dict):
            return False
        extracted_text = output.get("raw_text", "")
        return isinstance(extracted_text, str) and len(extracted_text.strip()) > 0

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text from the uploaded prescription file.

        Args:
            input_data: dict with keys file_bytes, mime_type, filename.

        Returns:
            Dict payload compatible with BaseAgent/Orchestrator chaining.
        """
        file_bytes: bytes = input_data["file_bytes"]
        mime_type: str = input_data.get("mime_type", "")
        filename: str = input_data.get("filename", "unknown")

        self.logger.info(
            f"Ingesting file: {filename!r} "
            f"({mime_type}, {len(file_bytes):,} bytes)"
        )

        text = PrescriptionParser.parse(file_bytes, mime_type, filename)

        self.logger.info(
            f"Extracted {len(text)} characters from {filename!r}"
        )
        # BaseAgent expects dict output and pops _confidence from it.
        # Include a minimal diagnosis hint so downstream PrescriptionAnalyzer
        # accepts dict input and can parse from raw_text.
        return {
            "raw_text": text,
            "diagnosis": "Unknown",
            "source_filename": filename,
            "source_mime_type": mime_type,
            "_confidence": 0.85,
        }
