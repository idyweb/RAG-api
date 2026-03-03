"""
PDF parsing utility with Markdown extraction.

Dual extraction strategy:
- Primary: pymupdf4llm → structure-preserving Markdown (headers, tables, lists)
- Fallback: pypdf → plain text (for corrupted or scanned PDFs)

The Markdown output enables hierarchical chunking downstream, which produces
significantly better RAG retrieval vs flat text chunking.
"""

from typing import IO, Tuple
import os
from pathlib import Path

# Primary: pymupdf4llm (Markdown output)
try:
    import pymupdf
    import pymupdf4llm

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Fallback: pypdf (plain text)
from pypdf import PdfReader

from api.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress tokenizer parallelism warning from pymupdf internals
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def extract_text_from_pdf(
    file_obj: IO[bytes], as_markdown: bool = True
) -> Tuple[str, str]:
    """
    Extract text from a PDF file stream.

    Args:
        file_obj: The uploaded file stream containing the PDF
        as_markdown: If True, extract as Markdown (preserves structure).
                     If False, extract as plain text.

    Returns:
        Tuple of (extracted_text, content_format).
        content_format is "markdown" or "text".

    Raises:
        ValueError: If PDF cannot be parsed by any method
    """
    if as_markdown and HAS_PYMUPDF:
        text = _extract_markdown(file_obj)
        if text:
            return text, "markdown"
        # If markdown extraction returned empty, fall through to plain text
        file_obj.seek(0)

    text = _extract_plain_text(file_obj)
    return text, "text"


def _extract_markdown(file_obj: IO[bytes]) -> str:
    """
    Extract PDF as Markdown using pymupdf4llm.

    Preserves headers, tables, lists, and structural formatting.
    Uses a temporary file because pymupdf requires a file path.

    Returns:
        Markdown string, or empty string on failure
    """
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name

        try:
            doc = pymupdf.open(tmp_path)

            markdown = pymupdf4llm.to_markdown(
                doc,
                page_chunks=False,
                write_images=False,
                image_path=None,
                image_format="png",
                dpi=150,
                page_width=None,
                page_height=None,
                margins=(0, 50, 0, 50),
                fontsize_limit=3,
                ignore_code=False,
                table_strategy="lines_strict",
            )

            # Clean encoding issues
            markdown_cleaned = markdown.encode(
                "utf-8", errors="surrogatepass"
            ).decode("utf-8", errors="ignore")

            logger.info(
                f"Extracted {len(markdown_cleaned)} chars as Markdown "
                f"from PDF ({len(doc)} pages)"
            )

            doc.close()
            return markdown_cleaned.strip()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.warning(
            f"pymupdf4llm extraction failed: {e}, falling back to plain text"
        )
        file_obj.seek(0)
        return ""


def _extract_plain_text(file_obj: IO[bytes]) -> str:
    """
    Extract PDF as plain text using pypdf (fallback).

    No structure preservation — purely sequential text extraction.

    Returns:
        Extracted text

    Raises:
        ValueError: If PDF cannot be parsed
    """
    try:
        reader = PdfReader(file_obj)
        text_parts = []

        for _page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n\n".join(text_parts).strip()

        logger.info(
            f"Extracted {len(full_text)} chars as plain text "
            f"from PDF ({len(reader.pages)} pages)"
        )

        return full_text

    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        raise ValueError(
            "Could not extract text from the provided PDF file. "
            "Ensure it is a valid format and not purely scanned images."
        )
