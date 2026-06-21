import logging
from pathlib import Path
from typing import Optional

from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import PdfContent

from .pymupdf_parser import PyMuPDFParser

logger = logging.getLogger(__name__)


class PDFParserService:
    """Main PDF parsing service using PyMuPDF as the PDF backend."""

    def __init__(self, max_pages: int = 20, max_file_size_mb: int = 20):
        """
        Initialize PDF parser service with configurable limits.

        Args:
            max_pages: Maximum number of pages to process (default: 20)
            max_file_size_mb: Maximum file size in MB (default: 20MB)
        """
        self._parser = PyMuPDFParser(max_pages=max_pages, max_file_size_mb=max_file_size_mb)

    async def parse_pdf(self, pdf_path: Path) -> Optional[PdfContent]:
        """
        Parse PDF using the PyMuPDF parser.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PdfContent object or None if parsing failed
        """
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise PDFValidationError(f"PDF file not found: {pdf_path}")

        try:
            result = await self._parser.parse_pdf(pdf_path)
            if result:
                logger.info(f"Parsed {pdf_path.name}")
                return result

            # The parser returns None to signal a deliberate skip (e.g. the file is
            # over the size limit), not a parse failure. Propagate None so the
            # caller stores the paper with metadata only instead of recording it
            # as a pipeline error.
            logger.info(f"Skipped PDF parsing for {pdf_path.name} (over limit); storing metadata only")
            return None

        except (PDFValidationError, PDFParsingException):
            raise
        except Exception as e:
            logger.error(f"PDF parsing error for {pdf_path.name}: {e}")
            raise PDFParsingException(f"PDF parsing error for {pdf_path.name}: {e}")
