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

    def _load_document(self, pdf_path: Path, max_pages: Optional[int] = None) -> tuple[object, int]:
        """Load a PDF, reading at most ``max_pages`` pages.

        ``py_pdf_parser.load_file`` reads the entire document into memory before
        returning, which is the root cause of the old size/page limits. We
        replicate its lightweight loader here so we can pass ``maxpages`` down to
        pdfminer and bound both memory use and processing time regardless of how
        long the paper is. Pages beyond ``max_pages`` are simply not read.

        Args:
            pdf_path: Path to the PDF file.
            max_pages: Maximum number of pages to read (``None``/0 means no limit).

        Returns:
            Tuple of (PDFDocument, number of pages actually read).
        """
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LAParams, LTFigure, LTTextBox
            from py_pdf_parser.components import PDFDocument
            from py_pdf_parser.loaders import DEFAULT_LA_PARAMS, Page
        except ImportError as exc:
            raise PDFParsingException(
                "py-pdf-parser is not installed. Install project dependencies to enable PDF parsing."
            ) from exc

        la_params = {**DEFAULT_LA_PARAMS, "all_texts": True}
        maxpages = max_pages or 0  # pdfminer treats 0 as "no limit"

        pages: dict = {}
        pages_read = 0
        with open(pdf_path, "rb") as in_file:
            for page in extract_pages(in_file, laparams=LAParams(**la_params), maxpages=maxpages):
                pages_read += 1
                elements = [element for element in page if isinstance(element, LTTextBox)]
                # With all_texts=True we also pull text nested inside figures.
                for figure in (element for element in page if isinstance(element, LTFigure)):
                    elements += [element for element in figure if isinstance(element, LTTextBox)]
                if not elements:
                    continue
                pages[page.pageid] = Page(width=page.width, height=page.height, elements=elements)

        return PDFDocument(pages=pages, pdf_file_path=str(pdf_path)), pages_read

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

    def _validate_pdf(self, pdf_path: Path) -> Optional[int]:
        """
        Validate the PDF and report its page count.

        Only hard failures raise ``PDFValidationError``: empty file, missing PDF
        header, or file size over the configured limit. The page count is
        *returned* rather than enforced, so that papers longer than ``max_pages``
        are truncated to the first ``max_pages`` pages (see :meth:`parse_pdf`)
        instead of being dropped entirely.

        Args:
            pdf_path: Path to PDF file

        Returns:
            The PDF page count, or ``None`` if it could not be determined.
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

            # Report (do not enforce) the page count; long papers are truncated, not skipped.
            return self._get_page_count(pdf_path)

        except PDFValidationError:
            raise
        except Exception as e:
            logger.error(f"Error validating PDF {pdf_path}: {e}")
            raise PDFValidationError(f"Error validating PDF {pdf_path}: {e}")

    async def parse_pdf(self, pdf_path: Path) -> Optional[PdfContent]:
        """
        Parse PDF using py-pdf-parser.

        Papers longer than ``max_pages`` are not skipped: the first ``max_pages``
        pages are parsed and the result is flagged as ``truncated`` (both as a
        first-class field on :class:`PdfContent` and inside ``metadata``) so the
        rest of the pipeline can record whether the stored content is complete.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PdfContent object or None if parsing was skipped (e.g. file too large)
        """
        try:
            # Validate PDF and learn its page count (does not enforce the page limit).
            pages_total = self._validate_pdf(pdf_path)

            # Warm up models on first use
            self._warm_up_models()

            document, pages_processed = self._load_document(pdf_path, max_pages=self.max_pages)
            elements = [element for element in document.elements if self._normalize_text(element.text())]
            raw_text = "\n".join(self._normalize_text(element.text()) for element in elements)
            sections = self._build_sections(elements)

            # The content is partial when the PDF had more pages than we read.
            truncated = pages_total is not None and pages_total > self.max_pages
            if truncated:
                logger.warning(
                    f"PDF {pdf_path.name} has {pages_total} pages; parsed first {pages_processed} "
                    f"(max_pages={self.max_pages}). Stored content is partial (truncated=True)."
                )

            # Focus on what arXiv API doesn't provide: structured full text content only.
            return PdfContent(
                sections=sections,
                figures=[],
                tables=[],
                raw_text=raw_text,
                references=[],
                parser_used=ParserType.PY_PDF_PARSER,
                truncated=truncated,
                pages_total=pages_total,
                pages_processed=pages_processed,
                metadata={
                    "source": "py-pdf-parser",
                    "note": "Content extracted from PDF, metadata comes from arXiv API",
                    "complete": not truncated,
                    "truncated": truncated,
                    "pages_total": pages_total,
                    "pages_processed": pages_processed,
                    "max_pages": self.max_pages,
                },
            )

        except PDFValidationError as e:
            # Handle file-size limit gracefully by returning None; page-count is no
            # longer a skip reason (long papers are truncated instead).
            error_msg = str(e).lower()
            if "too large" in error_msg:
                logger.info(f"Skipping PDF processing due to file-size limit: {e}")
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

            if "not valid" in error_msg:
                logger.error("PDF appears to be corrupted or not a valid PDF file")
                raise PDFParsingException(f"PDF appears to be corrupted or invalid: {pdf_path}")
            elif "timeout" in error_msg:
                logger.error("PDF processing timed out - file may be too complex")
                raise PDFParsingException(f"PDF processing timed out: {pdf_path}")
            elif "memory" in error_msg or "ram" in error_msg:
                logger.error("Out of memory - PDF may be too large or complex")
                raise PDFParsingException(f"Out of memory processing PDF: {pdf_path}")
            else:
                raise PDFParsingException(f"Failed to parse PDF with py-pdf-parser: {e}")
