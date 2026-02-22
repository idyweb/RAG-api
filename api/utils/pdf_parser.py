"""
PDF parsing utility.

Extracts text from PDF files using pypdf.
"""

from typing import IO
from pypdf import PdfReader
from api.utils.logger import get_logger

logger = get_logger(__name__)

def extract_text_from_pdf(file_obj: IO[bytes]) -> str:
    """
    Extract all text from a PDF file stream.
    
    Args:
        file_obj: The uploaded file stream containing the PDF (e.g. from FastAPI UploadFile.file)
        
    Returns:
        The full extracted text as a string
    """
    try:
        reader = PdfReader(file_obj)
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
                
        full_text = "\n".join(text_parts).strip()
        logger.info(f"Extracted {len(full_text)} characters from PDF ({len(reader.pages)} pages)")
        return full_text
        
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        raise ValueError("Could not extract text from the provided PDF file. Ensure it is a valid format and not purely flat images.")
