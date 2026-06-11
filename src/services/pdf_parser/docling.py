import logging
import re
from pathlib import Path
from typing import Optional
from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import PaperFigure, PaperSection, PaperTable, ParserType, PdfContent

logger = logging.getLogger(__name__)


class DoclingParser:
    """PDF parser backed by py-pdf-parser for fallback when GROBID fails."""

    def __init__(self, max_pages: int = 20, max_file_size_mb: int = 20, do_ocr: bool = False, do_table_structure: bool = True):
        """
        Initialize the parser with limits kept for compatibility with the old Docling interface.

        Args:
            max_pages: Maximum number of pages to process (default: 20)
            max_file_size_mb: Maximum file size in MB (default: 20MB)
            do_ocr: Kept for interface compatibility; unused by py-pdf-parser.
            do_table_structure: Kept for interface compatibility; unused by py-pdf-parser.
        """
        self._warmed_up = False
        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure
        self._warned_missing_pdfium = False

    def _warm_up_models(self):
        """Pre-warm the models with a small dummy document to avoid cold start."""
        if not self._warmed_up:
            # This happens only once per parser instance.
            self._warmed_up = True

    def _load_document(self, pdf_path: Path):
        """Load a PDF document through py-pdf-parser."""
        try:
            from py_pdf_parser.loaders import load_file
        except ImportError as exc:
            raise PDFParsingException(
                "py-pdf-parser is not installed. Install project dependencies to enable PDF parsing."
            ) from exc

        return load_file(str(pdf_path), la_params={"all_texts": True})

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse repeated whitespace, strip NUL bytes, and trim the text.

        PDF extractors (pdfminer) can emit NUL (0x00) characters that PostgreSQL
        rejects in text columns, so they are removed here before any storage.
        """
        text = text.replace("\x00", "")
        return re.sub(r"\s+", " ", text).strip()

    def _looks_like_heading(self, text: str, element: object, is_first_element: bool = False) -> bool:
        """Heuristic heading detector used to build sections without Docling structure."""
        normalized = self._normalize_text(text)
        if not normalized:
            return False

        lowered = normalized.lower().rstrip(":")
        if lowered in {
            "abstract",
            "introduction",
            "related work",
            "background",
            "method",
            "methods",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "conclusions",
            "references",
            "acknowledgements",
            "acknowledgments",
        }:
            return True

        if re.match(r"^\d+(?:\.\d+)*\s+\S+", normalized):
            return True

        if normalized.endswith(":") and len(normalized) <= 80:
            return True

        if len(normalized.split()) <= 12 and len(normalized) <= 120:
            font_name = getattr(element, "font", "") or ""
            font_name = str(font_name).lower()
            if any(token in font_name for token in ("bold", "black", "semibold", "medium")):
                return True
            if is_first_element and len(normalized) <= 90:
                return True

        return False

    def _build_sections(self, elements: list[object]) -> list[PaperSection]:
        """Convert a flat element stream into paper sections."""
        sections: list[PaperSection] = []
        current_title = "Content"
        current_lines: list[str] = []

        def flush_section() -> None:
            if current_lines:
                sections.append(PaperSection(title=current_title, content="\n".join(current_lines).strip()))

        for index, element in enumerate(elements):
            element_text = self._normalize_text(element.text())
            if not element_text:
                continue

            if self._looks_like_heading(element_text, element, is_first_element=index == 0 and not sections and not current_lines):
                flush_section()
                current_title = element_text.rstrip(":")
                current_lines = []
                continue

            current_lines.append(element_text)

        flush_section()

        if not sections and current_lines:
            sections.append(PaperSection(title="Content", content="\n".join(current_lines).strip()))

        return sections

    def _get_page_count(self, pdf_path: Path) -> Optional[int]:
        """
        Return the number of pages when an optional PDF backend is available.

        Page-count validation is treated as best-effort so PDF parsing still
        works in environments that only install `py-pdf-parser`.
        """
        try:
            import pypdfium2 as pdfium
        except ImportError:
            if not self._warned_missing_pdfium:
                logger.warning("pypdfium2 is not installed; skipping PDF page-count validation")
                self._warned_missing_pdfium = True
            return None

        pdf_doc = pdfium.PdfDocument(str(pdf_path))
        try:
            return len(pdf_doc)
        finally:
            pdf_doc.close()

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """
        Comprehensive PDF validation including size and page limits.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if PDF appears valid and within limits, False otherwise
        """
        try:
            # Check file exists and is not empty
            if pdf_path.stat().st_size == 0:
                logger.error(f"PDF file is empty: {pdf_path}")
                raise PDFValidationError(f"PDF file is empty: {pdf_path}")

            # Check file size limit
            file_size = pdf_path.stat().st_size
            if file_size > self.max_file_size_bytes:
                logger.warning(
                    f"PDF file size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({self.max_file_size_bytes / 1024 / 1024:.1f}MB), skipping processing"
                )
                raise PDFValidationError(
                    f"PDF file too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size_bytes / 1024 / 1024:.1f}MB"
                )

            # Check if file starts with PDF header
            with open(pdf_path, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    logger.error(f"File does not have PDF header: {pdf_path}")
                    raise PDFValidationError(f"File does not have PDF header: {pdf_path}")

            # Check page count limit when an optional backend is available.
            actual_pages = self._get_page_count(pdf_path)
            if actual_pages is not None and actual_pages > self.max_pages:
                logger.warning(
                    f"PDF has {actual_pages} pages, exceeding limit of {self.max_pages} pages. Skipping processing to avoid performance issues."
                )
                raise PDFValidationError(f"PDF has too many pages: {actual_pages} > {self.max_pages}")

            return True

        except PDFValidationError:
            raise
        except Exception as e:
            logger.error(f"Error validating PDF {pdf_path}: {e}")
            raise PDFValidationError(f"Error validating PDF {pdf_path}: {e}")

    async def parse_pdf(self, pdf_path: Path) -> Optional[PdfContent]:
        """
        Parse PDF using py-pdf-parser as fallback parser.
        Limited to 20 pages to avoid memory issues with large papers.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PdfContent object or None if parsing failed
        """
        try:
            # Validate PDF first (includes size and page limits)
            self._validate_pdf(pdf_path)

            # Warm up models on first use
            self._warm_up_models()

            document = self._load_document(pdf_path)
            elements = [element for element in document.elements if self._normalize_text(element.text())]
            raw_text = "\n".join(self._normalize_text(element.text()) for element in elements)
            sections = self._build_sections(elements)

            # Focus on what arXiv API doesn't provide: structured full text content only.
            return PdfContent(
                sections=sections,
                figures=[],
                tables=[],
                raw_text=raw_text,
                references=[],
                parser_used=ParserType.PY_PDF_PARSER,
                metadata={"source": "py-pdf-parser", "note": "Content extracted from PDF, metadata comes from arXiv API"},
            )

        except PDFValidationError as e:
            # Handle size/page limit validation errors gracefully by returning None
            error_msg = str(e).lower()
            if "too large" in error_msg or "too many pages" in error_msg:
                logger.info(f"Skipping PDF processing due to size/page limits: {e}")
                return None
            else:
                # Re-raise other validation errors (corrupted files, etc.)
                raise
        except Exception as e:
            logger.error(f"Failed to parse PDF with py-pdf-parser: {e}")
            logger.error(f"PDF path: {pdf_path}")
            logger.error(f"PDF size: {pdf_path.stat().st_size} bytes")
            logger.error(f"Error type: {type(e).__name__}")

            # Add specific handling for common issues
            error_msg = str(e).lower()

            # Note: Page and size limit checks are now handled in _validate_pdf method

            if "not valid" in error_msg:
                logger.error("PDF appears to be corrupted or not a valid PDF file")
                raise PDFParsingException(f"PDF appears to be corrupted or invalid: {pdf_path}")
            elif "timeout" in error_msg:
                logger.error("PDF processing timed out - file may be too complex")
                raise PDFParsingException(f"PDF processing timed out: {pdf_path}")
            elif "memory" in error_msg or "ram" in error_msg:
                logger.error("Out of memory - PDF may be too large or complex")
                raise PDFParsingException(f"Out of memory processing PDF: {pdf_path}")
            elif "max_num_pages" in error_msg or "page" in error_msg:
                logger.error(f"PDF processing issue likely related to page limits (current limit: {self.max_pages} pages)")
                raise PDFParsingException(
                    f"PDF processing failed, possibly due to page limit ({self.max_pages} pages). Error: {e}"
                )
            else:
                raise PDFParsingException(f"Failed to parse PDF with py-pdf-parser: {e}")
