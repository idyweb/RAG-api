"""
PDF parsing utility with Markdown extraction.

Uses pymupdf4llm + pymupdf.layout for structure-preserving Markdown output
(headers, tables, lists, reading order). The Markdown output enables
hierarchical chunking downstream, producing significantly better RAG
retrieval vs flat text chunking.
"""

from typing import IO
import os
from pathlib import Path

import pymupdf
import pymupdf.layout  # Activates improved page layout analysis
import pymupdf4llm

from api.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress tokenizer parallelism warning from pymupdf internals
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def extract_text_from_pdf(file_obj: IO[bytes]) -> str:
    """
    Extract text from a PDF file stream as Markdown.

    Uses pymupdf4llm with the layout engine for structure-preserving
    extraction (headers, tables, lists, column detection, reading order).

    Args:
        file_obj: The uploaded file stream containing the PDF

    Returns:
        Extracted Markdown text

    Raises:
        ValueError: If PDF cannot be parsed
    """
    import tempfile

    try:
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
            text = markdown_cleaned.strip()

            if not text:
                raise ValueError(
                    "No readable text found in the PDF. "
                    "The file may be purely scanned images."
                )

            return text

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise ValueError(
            "Could not extract text from the provided PDF file. "
            "Ensure it is a valid format and not purely scanned images."
        )
