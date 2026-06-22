import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import PaperSection, ParserType, PdfContent

logger = logging.getLogger(__name__)

# PyMuPDF span flag bit for bold text (see fitz TEXT_FONT_* constants).
_FITZ_BOLD_FLAG = 1 << 4


@dataclass
class _TextLine:
    """A single extracted line of text plus the typographic signals we use to
    tell headings apart from body text (font size and weight)."""

    text: str
    size: float
    bold: bool


class PyMuPDFParser:
    """PDF parser backed by PyMuPDF (fitz).

    Chosen for being fast (C-based), lightweight, and for exposing reliable
    per-span font metrics that let us detect section headings far more
    accurately than font-name heuristics.
    """

    def __init__(self, max_pages: int = 20, max_file_size_mb: int = 20):
        """
        Initialize the parser with size/page limits.

        Args:
            max_pages: Maximum number of pages to process (default: 20)
            max_file_size_mb: Maximum file size in MB (default: 20MB)
        """
        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    @staticmethod
    def _open_document(pdf_path: Path):
        """Open a PDF with PyMuPDF, raising a clear error if the backend is missing."""
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise PDFParsingException(
                "PyMuPDF (pymupdf) is not installed. Install project dependencies to enable PDF parsing."
            ) from exc

        return fitz.open(str(pdf_path))

    def _load_document(self, pdf_path: Path, max_pages: Optional[int] = None) -> tuple[list[_TextLine], int]:
        """Load a PDF with PyMuPDF, reading at most ``max_pages`` pages.

        PyMuPDF reads pages lazily, so we only ever touch the first ``max_pages``
        pages; the rest of a long paper is never decoded. Text is pulled per line
        (``get_text("dict")``) in the document's content-stream order, which is the
        natural reading order for well-formed arXiv PDFs.

        Args:
            pdf_path: Path to the PDF file.
            max_pages: Maximum number of pages to read (``None``/0 means no limit).

        Returns:
            Tuple of (list of text lines, number of pages actually read).
        """
        document = self._open_document(pdf_path)
        try:
            pages_total = document.page_count
            limit = min(pages_total, max_pages) if max_pages else pages_total

            lines: list[_TextLine] = []
            for page_index in range(limit):
                page = document.load_page(page_index)
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    # Image/drawing blocks have no "lines" key; skip them.
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        text = self._normalize_text("".join(span.get("text", "") for span in spans))
                        if not text:
                            continue
                        size = max((span.get("size", 0.0) for span in spans), default=0.0)
                        bold = any(self._span_is_bold(span) for span in spans)
                        lines.append(_TextLine(text=text, size=round(size, 1), bold=bold))

            return lines, limit
        finally:
            document.close()

    @staticmethod
    def _span_is_bold(span: dict) -> bool:
        """Detect bold text from PyMuPDF span flags, with a font-name fallback."""
        if span.get("flags", 0) & _FITZ_BOLD_FLAG:
            return True
        font_name = str(span.get("font", "")).lower()
        return any(token in font_name for token in ("bold", "black", "semibold"))

    def _extract_content(self, pdf_path: Path) -> tuple[str, list[PaperSection], int]:
        """Synchronous, CPU-bound text extraction.

        PDF parsing and section building are blocking and CPU-heavy, so this runs
        in a worker thread (see :meth:`parse_pdf`) to keep the event loop free for
        concurrent downloads.

        Returns:
            Tuple of (raw_text, sections, number of pages actually read).
        """
        lines, pages_processed = self._load_document(pdf_path, max_pages=self.max_pages)
        raw_text = "\n".join(line.text for line in lines)
        sections = self._build_sections(lines)
        return raw_text, sections, pages_processed

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse repeated whitespace, strip NUL bytes, and trim the text.

        PDF extractors can emit NUL (0x00) characters that PostgreSQL rejects in
        text columns, so they are removed here before any storage.
        """
        text = text.replace("\x00", "")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _estimate_body_size(lines: list[_TextLine]) -> float:
        """Estimate the body-text font size as the most common line size.

        Headings are then anything meaningfully larger than this baseline, which
        is far more robust than matching hard-coded font names.
        """
        counts = Counter(round(line.size) for line in lines if line.size)
        if not counts:
            return 0.0
        return float(counts.most_common(1)[0][0])

    def _looks_like_heading(self, line: _TextLine, body_size: float, is_first_line: bool = False) -> bool:
        """Heuristic heading detector combining known titles, numbering, and font size."""
        normalized = line.text
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

        # Numbered sections: "1 Introduction", "1.2 Background", etc.
        if re.match(r"^\d+(?:\.\d+)*\s+\S+", normalized):
            return True

        if normalized.endswith(":") and len(normalized) <= 80:
            return True

        word_count = len(normalized.split())

        # A line set in a noticeably larger font than the body text, and short
        # enough to be a title rather than a sentence, is treated as a heading.
        if body_size and line.size >= body_size * 1.10 and word_count <= 16 and len(normalized) <= 140:
            return True

        if word_count <= 12 and len(normalized) <= 120:
            if line.bold:
                return True
            if is_first_line and len(normalized) <= 90:
                return True

        return False

    def _build_sections(self, lines: list[_TextLine]) -> list[PaperSection]:
        """Convert a flat line stream into paper sections."""
        sections: list[PaperSection] = []
        current_title = "Content"
        current_lines: list[str] = []
        body_size = self._estimate_body_size(lines)

        def flush_section() -> None:
            if current_lines:
                sections.append(PaperSection(title=current_title, content="\n".join(current_lines).strip()))

        for index, line in enumerate(lines):
            if not line.text:
                continue

            is_first = index == 0 and not sections and not current_lines
            if self._looks_like_heading(line, body_size, is_first_line=is_first):
                flush_section()
                current_title = line.text.rstrip(":")
                current_lines = []
                continue

            current_lines.append(line.text)

        flush_section()

        if not sections and current_lines:
            sections.append(PaperSection(title="Content", content="\n".join(current_lines).strip()))

        return sections

    def _get_page_count(self, pdf_path: Path) -> Optional[int]:
        """Return the number of pages using PyMuPDF."""
        document = self._open_document(pdf_path)
        try:
            return document.page_count
        finally:
            document.close()

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
        Parse PDF using PyMuPDF.

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

            # Offload the blocking, CPU-bound extraction to a worker thread so the
            # event loop stays free and concurrent PDF downloads keep progressing.
            raw_text, sections, pages_processed = await asyncio.to_thread(self._extract_content, pdf_path)

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
                parser_used=ParserType.PYMUPDF,
                truncated=truncated,
                pages_total=pages_total,
                pages_processed=pages_processed,
                metadata={
                    "source": "pymupdf",
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
            logger.error(f"Failed to parse PDF with PyMuPDF: {e}")
            logger.error(f"PDF path: {pdf_path}")
            logger.error(f"PDF size: {pdf_path.stat().st_size} bytes")
            logger.error(f"Error type: {type(e).__name__}")

            # Add specific handling for common issues
            error_msg = str(e).lower()

            if "not valid" in error_msg or "cannot open" in error_msg or "no objects found" in error_msg:
                logger.error("PDF appears to be corrupted or not a valid PDF file")
                raise PDFParsingException(f"PDF appears to be corrupted or invalid: {pdf_path}")
            elif "timeout" in error_msg:
                logger.error("PDF processing timed out - file may be too complex")
                raise PDFParsingException(f"PDF processing timed out: {pdf_path}")
            elif "memory" in error_msg or "ram" in error_msg:
                logger.error("Out of memory - PDF may be too large or complex")
                raise PDFParsingException(f"Out of memory processing PDF: {pdf_path}")
            else:
                raise PDFParsingException(f"Failed to parse PDF with PyMuPDF: {e}")
