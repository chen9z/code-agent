from __future__ import annotations

"""Tree-sitter backed semantic splitter shared by retrieval pipelines."""

import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tree_sitter import Parser

from adapters.workspace.tree_sitter.language_loader import (
    is_language_supported,
    load_language,
)

logger = logging.getLogger(__name__)

EXT_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cs": "c_sharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
}

CODE_EXTS = set(EXT_LANGUAGE.keys())

_NODE_TYPES: Dict[str, Sequence[str]] = {
    "python": ["function_definition", "class_definition"],
    "javascript": ["function_declaration", "method_definition", "class_declaration"],
    "typescript": [
        "function_declaration",
        "method_signature",
        "method_definition",
        "class_declaration",
    ],
    "tsx": ["function_declaration", "method_definition", "class_declaration"],
    "go": ["function_declaration", "method_declaration"],
    "java": ["method_declaration", "class_declaration"],
    "rust": ["function_item", "impl_item"],
    "c": ["function_definition"],
    "cpp": ["function_definition"],
    "c_sharp": ["method_declaration", "class_declaration"],
    "php": ["function_definition", "method_declaration", "class_declaration"],
    "ruby": ["method", "class"],
    "swift": ["function_declaration", "class_declaration"],
    "kotlin": ["function_declaration", "class_declaration"],
}

_PARSER_CACHE: Dict[str, Parser] = {}


def _node_types(language: str) -> Sequence[str]:
    return _NODE_TYPES.get(language, ())


class SpanKind(Enum):
    CODE = auto()
    COMMENT = auto()


@dataclass(slots=True)
class Document:
    path: str
    content: str
    start_line: int
    end_line: int
    language: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Span:
    start: int
    end: int
    kind: SpanKind = SpanKind.CODE


def _get_parser(language: str) -> Optional[Parser]:
    if not is_language_supported(language):
        return None
    parser = _PARSER_CACHE.get(language)
    if parser is None:
        try:
            parser = Parser()
            parser.language = load_language(language)
            _PARSER_CACHE[language] = parser
        except Exception as exc:  # pragma: no cover - defensive log
            logger.warning("Failed to load tree-sitter language '%s': %s", language, exc)
            return None
    return parser


def _byte_offsets(lines: Sequence[bytes]) -> List[int]:
    offsets = [0]
    total = 0
    for line in lines:
        total += len(line)
        offsets.append(total)
    return offsets


def _byte_to_line(line_offsets: Sequence[int], byte_idx: int, total_lines: int) -> int:
    pos = bisect_right(line_offsets, byte_idx) - 1
    return max(1, min(total_lines, pos))


def _is_comment_node(node) -> bool:
    node_type = getattr(node, "type", "") or ""
    return "comment" in node_type.lower()


def _get_node_name(node) -> Optional[str]:
    if node is None:
        return None
    try:
        target = node.child_by_field_name("name")
        if target is None:
            return None
        return target.text.decode("utf-8", errors="ignore")
    except Exception:  # pragma: no cover
        return None


class SemanticSplitter:
    def __init__(self, language: str, chunk_size: int = 2048) -> None:
        self.language = language
        self.chunk_size = chunk_size

    def split(self, path: str, text: bytes) -> List[Document]:
        if not text:
            return []
        
        lines = text.splitlines(keepends=True)
        line_offsets = _byte_offsets(lines)
        
        parser = _get_parser(self.language)
        if parser is None:
            return self._fallback_chunks(path, text, lines, line_offsets)
            
        try:
            tree = parser.parse(text)
        except Exception as exc:
            logger.warning("tree-sitter parse failure for %s (%s)", path, exc)
            return self._fallback_chunks(path, text, lines, line_offsets)
            
        root = tree.root_node
        if root is None:
            return self._fallback_chunks(path, text, lines, line_offsets)
            
        spans = self._chunk_node(root)
        self._connect_chunks(spans)
        spans = self._coalesce_chunks(spans)
        
        if not spans:
            return self._fallback_chunks(path, text, lines, line_offsets)
            
        return self._build_chunks_from_spans(
            path=path,
            encoded=text,
            lines=lines,
            spans=spans,
            line_offsets=line_offsets,
        )

    def _fallback_chunks(
            self,
            path: str,
            text: bytes,
            lines: Sequence[bytes],
            line_offsets: Sequence[int],
    ) -> List[Document]:
        size = max(1, self.chunk_size)
        total_bytes = len(text)
        start_byte = 0
        chunks: List[Document] = []
        
        while start_byte < total_bytes:
            end_byte = min(total_bytes, start_byte + size)
            # Ensure we don't split in the middle of a multi-byte character if possible
            # But for fallback simple byte slicing is acceptable or we can align to utf-8
            # For simplicity here we just slice bytes, decoding will handle errors
            
            content_bytes = text[start_byte:end_byte]
            content = content_bytes.decode("utf-8", errors="replace")
            
            if content.strip():
                chunks.append(
                    Document(
                        path=path,
                        content=content,
                        start_line=_byte_to_line(line_offsets, start_byte, len(lines)),
                        end_line=_byte_to_line(line_offsets, end_byte, len(lines)),
                        language=self.language,
                        metadata={"kind": SpanKind.CODE.name},
                    )
                )
            start_byte = end_byte
        return chunks

    def _chunk_node(self, node) -> List[Span]:
        span = Span(node.start_byte, node.start_byte)
        chunks: List[Span] = []
        symbol_types = set(_node_types(self.language))
        pending_comment: Optional[List[int]] = None

        def flush_pending_comment() -> None:
            nonlocal pending_comment
            if pending_comment and pending_comment[1] > pending_comment[0]:
                chunks.append(Span(pending_comment[0], pending_comment[1], SpanKind.COMMENT))
            pending_comment = None

        for child in node.children:
            if _is_comment_node(child):
                if pending_comment is None:
                    pending_comment = [child.start_byte, child.end_byte]
                else:
                    pending_comment[1] = child.end_byte
                continue

            comment_range: Optional[Tuple[int, int]] = None
            if pending_comment is not None:
                if not symbol_types or child.type in symbol_types:
                    comment_range = (pending_comment[0], pending_comment[1])
                    pending_comment = None
                else:
                    flush_pending_comment()

            child_start = comment_range[0] if comment_range else child.start_byte
            child_end = child.end_byte
            child_len = child_end - child_start

            if comment_range and span.end > span.start:
                chunks.append(span)
                span = Span(comment_range[0], comment_range[0])

            if child_len > self.chunk_size:
                if span.end > span.start:
                    chunks.append(span)
                span = Span(child_end, child_end)
                if len(child.children) == 0:
                    start = comment_range[0] if comment_range else child.start_byte
                    chunks.append(Span(start, child_end))
                    continue

                child_spans = self._chunk_node(child)
                if comment_range and child_spans:
                    child_spans[0].start = min(child_spans[0].start, comment_range[0])
                elif comment_range:
                    chunks.append(Span(comment_range[0], comment_range[1], SpanKind.COMMENT))
                chunks.extend(child_spans)
                continue

            if (child_end - span.start) > self.chunk_size:
                if span.end > span.start:
                    chunks.append(span)
                start = comment_range[0] if comment_range else child.start_byte
                span = Span(start, child_end)
            else:
                if span.end == span.start and comment_range:
                    span = Span(comment_range[0], child_end)
                else:
                    span = Span(span.start, child_end)

        flush_pending_comment()

        if span.end > span.start:
            chunks.append(span)
        return chunks

    def _connect_chunks(self, chunks: List[Span]):
        for pre, cur in zip(chunks[:-1], chunks[1:]):
            pre.end = cur.start

    def _coalesce_chunks(self, chunks: List[Span]) -> List[Span]:
        new_chunks = []
        current_chunk = Span(0, 0)
        for chunk in chunks:
            if (chunk.end - current_chunk.start) < self.chunk_size:
                current_chunk.end = chunk.end
            else:
                if current_chunk.end > current_chunk.start:
                    new_chunks.append(current_chunk)
                current_chunk = Span(chunk.start, chunk.end, chunk.kind)
        if current_chunk.end > current_chunk.start:
            new_chunks.append(current_chunk)
        return new_chunks

    def _build_chunks_from_spans(
            self,
            path: str,
            encoded: bytes,
            lines: Sequence[bytes],
            spans: List[Span],
            line_offsets: Sequence[int],
    ) -> List[Document]:
        chunks: List[Document] = []
        total_lines = len(lines)
        for span in spans:
            content_bytes = encoded[span.start:span.end]
            content = content_bytes.decode("utf-8", errors="replace")
            if not content.strip():
                continue
            chunk = Document(
                path=path,
                content=content,
                start_line=_byte_to_line(line_offsets, span.start, total_lines),
                end_line=_byte_to_line(line_offsets, span.end, total_lines),
                language=self.language,
                metadata={"kind": span.kind.name if span.kind else SpanKind.CODE.name},
            )
            chunks.append(chunk)
        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks
