import logging
from pathlib import Path
from typing import Optional

from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import PdfContent

from .docling import DoclingParser

logger = logging.getLogger(__name__)


class PDFParserService:
    """Main PDF parsing service using py-pdf-parser as the PDF backend."""

    def __init__(self, max_pages: int = 20, max_file_size_mb: int = 20, do_ocr: bool = False, do_table_structure: bool = True):
        """
        Initialize PDF parser service with configurable limits.

        Args:
            max_pages: Maximum number of pages to process (default: 20)
            max_file_size_mb: Maximum file size in MB (default: 20MB)
            do_ocr: Enable OCR for scanned PDFs (default: False, very slow)
            do_table_structure: Extract table structures (default: True)
        """
        self.docling_parser = DoclingParser(
            max_pages=max_pages, max_file_size_mb=max_file_size_mb, do_ocr=do_ocr, do_table_structure=do_table_structure
        )

    async def parse_pdf(self, pdf_path: Path) -> Optional[PdfContent]:
        """
        Parse PDF using Docling parser only.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PdfContent object or None if parsing failed
        """
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise PDFValidationError(f"PDF file not found: {pdf_path}")

        try:
            result = await self.docling_parser.parse_pdf(pdf_path)
            if result:
                logger.info(f"Parsed {pdf_path.name}")
                return result

            # docling returns None to signal a deliberate skip (e.g. the file is
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
