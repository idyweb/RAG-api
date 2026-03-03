"""
Custom Markdown splitter with zero external dependencies.

Splits Markdown documents by headers while preserving document structure.
Used for hierarchical chunking in the RAG pipeline.

Why custom instead of LangChain:
- Zero transitive dependencies (no langchain-core bloat)
- Full control over edge cases (code blocks, ATX headers)
- Immune to upstream API breaking changes
- ~100 lines vs ~50 transitive packages
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MarkdownChunk:
    """Represents a chunk of Markdown with structural metadata."""

    content: str
    level: int  # Header level (1-6), 0 for preamble text
    header: str  # Header text (empty for preamble)
    start_line: int
    end_line: int


# Pre-compiled regex: matches ATX headers (# to ######) with required text.
# Negative lookbehind prevents matching inside code fences.
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# Detects code fence boundaries (``` or ~~~).
_CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


class MarkdownHeaderSplitter:
    """
    Split Markdown by headers (H1–H6).

    Unlike LangChain's MarkdownHeaderTextSplitter, this:
    - Has zero external dependencies
    - Correctly ignores '#' inside fenced code blocks
    - Returns stable string output (not Document objects that change between versions)
    - Is fully unit-testable with no mocking
    """

    def __init__(
        self,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
        strip_headers: bool = False,
    ) -> None:
        """
        Initialize splitter.

        Args:
            headers_to_split_on: List of (markdown_marker, level_name).
                Example: [("#", "H1"), ("##", "H2"), ("###", "H3")]
                Only headers at these levels will trigger splits.
            strip_headers: If True, remove header lines from chunk content.
        """
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
                ("####", "H4"),
                ("#####", "H5"),
                ("######", "H6"),
            ]

        self.strip_headers = strip_headers

        # Build a set of allowed header levels (e.g., {1, 2, 3})
        self._allowed_levels: set[int] = set()
        for marker, _level_name in headers_to_split_on:
            self._allowed_levels.add(len(marker))

    def split_text(self, text: str) -> List[str]:
        """
        Split Markdown text by headers.

        Returns:
            List of text chunks as plain strings.
            Empty list if input is empty/whitespace-only.
        """
        if not text or not text.strip():
            return []

        chunks = self._split_by_headers(text)
        return [chunk.content for chunk in chunks if chunk.content.strip()]

    def _split_by_headers(self, text: str) -> List[MarkdownChunk]:
        """
        Internal: Split text and return structured MarkdownChunk objects.

        Correctly handles:
        - Fenced code blocks (``` and ~~~) — '#' inside not treated as headers
        - Preamble text before any header
        - Consecutive headers with no body
        """
        lines = text.split("\n")
        chunks: List[MarkdownChunk] = []
        current_lines: List[str] = []
        current_header = ""
        current_level = 0
        chunk_start = 0
        in_code_fence = False

        for line_num, line in enumerate(lines):
            # Track code fence state to avoid splitting inside code blocks
            fence_match = _CODE_FENCE_RE.match(line.strip())
            if fence_match:
                in_code_fence = not in_code_fence
                current_lines.append(line)
                continue

            if in_code_fence:
                # Inside a code block — never split here
                current_lines.append(line)
                continue

            # Check if this line is a header we should split on
            header_match = self._match_header(line)

            if header_match is not None:
                level, header_text = header_match

                # Save previous chunk if it has content
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        chunks.append(
                            MarkdownChunk(
                                content=content,
                                level=current_level,
                                header=current_header,
                                start_line=chunk_start,
                                end_line=line_num - 1,
                            )
                        )

                # Start new chunk
                current_header = header_text
                current_level = level
                chunk_start = line_num

                if self.strip_headers:
                    current_lines = []
                else:
                    current_lines = [line]
            else:
                current_lines.append(line)

        # Save final chunk
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                chunks.append(
                    MarkdownChunk(
                        content=content,
                        level=current_level,
                        header=current_header,
                        start_line=chunk_start,
                        end_line=len(lines) - 1,
                    )
                )

        return chunks

    def _match_header(self, line: str) -> Optional[Tuple[int, str]]:
        """
        Check if a line is a Markdown ATX header that we should split on.

        Returns:
            (level, header_text) if it's a splittable header, None otherwise.
        """
        match = _HEADER_RE.match(line.strip())
        if match:
            level = len(match.group(1))
            if level in self._allowed_levels:
                return (level, match.group(2).strip())
        return None


def split_markdown_by_headers(
    text: str,
    headers: Optional[List[Tuple[str, str]]] = None,
    strip_headers: bool = False,
) -> List[str]:
    """
    Convenience function: Split Markdown by headers.

    Args:
        text: Markdown content to split
        headers: Header levels to split on (default: H1, H2, H3)
        strip_headers: Remove header lines from output chunks

    Returns:
        List of text chunks (strings). Empty list if input is empty.

    Example:
        >>> text = "# Title\\nContent\\n## Section\\nMore content"
        >>> chunks = split_markdown_by_headers(text)
        >>> len(chunks)
        2
    """
    splitter = MarkdownHeaderSplitter(
        headers_to_split_on=headers,
        strip_headers=strip_headers,
    )
    return splitter.split_text(text)
