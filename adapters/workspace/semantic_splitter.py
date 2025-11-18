from __future__ import annotations

"""Tree-sitter backed semantic splitter shared by retrieval pipelines."""

import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence

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
    kind: SpanKind

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


def _byte_offsets(lines: Sequence[str]) -> List[int]:
    offsets = [0]
    total = 0
    for line in lines:
        total += len(line.encode("utf-8"))
        offsets.append(total)
    return offsets


def _char_byte_offsets(text: str) -> List[int]:
    offsets = [0]
    total = 0
    for ch in text:
        total += len(ch.encode("utf-8"))
        offsets.append(total)
    return offsets


def _char_index(offsets: Sequence[int], byte_idx: int) -> int:
    return max(0, bisect_right(offsets, byte_idx) - 1)


def _char_length(offsets: Sequence[int], start: int, end: int) -> int:
    if end <= start:
        return 0
    return _char_index(offsets, end) - _char_index(offsets, start)


def _char_to_byte(offsets: Sequence[int], idx: int) -> int:
    if idx <= 0:
        return 0
    if idx >= len(offsets):
        return offsets[-1]
    return offsets[idx]


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
    def __init__(self, language: str, *, chunk_size: int = 2048) -> None:
        self.language = language
        self.chunk_size = chunk_size

    def split(self, path: str, text: str) -> List[Document]:
        text = text or ""
        if not text:
            return []
        lines = text.splitlines(keepends=True)
        encoded = text.encode("utf-8")
        line_offsets = _byte_offsets(lines)
        char_offsets = _char_byte_offsets(text)
        parser = _get_parser(self.language)
        if parser is None:
            return self._fallback_chunks(path, text, lines, line_offsets, char_offsets)
        try:
            tree = parser.parse(encoded)
        except Exception as exc:
            logger.warning("tree-sitter parse failure for %s (%s)", path, exc)
            return self._fallback_chunks(path, text, lines, line_offsets, char_offsets)
        root = tree.root_node
        if root is None:
            return self._fallback_chunks(path, text, lines, line_offsets, char_offsets)
        spans = self._chunk_node(root, text)
        self._connect_chunks(spans)
        spans = self._coalesce_chunks(spans)
        if not spans:
            return self._fallback_chunks(path, text, lines, line_offsets, char_offsets)
        return self._build_chunks_from_spans(
            path=path,
            encoded=encoded,
            lines=lines,
            spans=spans,
            line_offsets=line_offsets,
        )

    def _fallback_chunks(
            self,
            path: str,
            text: str,
            lines: Sequence[str],
            line_offsets: Sequence[int],
            char_offsets: Sequence[int],
    ) -> List[Document]:
        size = max(1, self.chunk_size)
        total_chars = len(char_offsets) - 1
        start_char = 0
        chunks: List[Document] = []
        while start_char < total_chars:
            end_char = min(total_chars, start_char + size)
            start_byte = _char_to_byte(char_offsets, start_char)
            end_byte = _char_to_byte(char_offsets, end_char)
            content = text[start_char:end_char]
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
            start_char = end_char
        return chunks

    def _chunk_node(self, node, text) -> List[Span]:
        span = Span(node.start_byte, node.start_byte)
        chunks = []
        for child in node.children:
            if child.end_byte - child.start_byte > self.chunk_size:
                if span.end > span.start:
                    chunks.append(span)
                span = Span(child.end_byte, child.end_byte)
                if len(child.children) == 0:
                    chunks.append(Span(child.start_byte, child.end_byte))
                else:
                    chunks.extend(self._chunk_node(child, text))
            elif child.end_byte - span.start > self.chunk_size:
                chunks.append(span)
                span = Span(child.start_byte, child.end_byte)
            else:
                span = Span(span.start, child.end_byte)

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
            if chunk.end - current_chunk.start < self.chunk_size:
                current_chunk.end = chunk.end
            else:
                if current_chunk.end > current_chunk.start:
                    new_chunks.append(current_chunk)
                current_chunk = Span(chunk.start, chunk.end)
        if current_chunk.end > current_chunk.start:
            new_chunks.append(current_chunk)
        return new_chunks

    def _build_chunks_from_spans(
            self,
            path: str,
            encoded: bytes,
            lines: Sequence[str],
            spans: List[Span],
            line_offsets: Sequence[int],
    ) -> List[Document]:
        chunks: List[Document] = []
        total_lines = len(lines)
        for span in spans:
            content_bytes = encoded[span.start:span.end]
            content = content_bytes.decode("utf-8", errors="ignore")
            if not content.strip():
                continue
            chunk = Document(
                path=path,
                content=content,
                start_line=_byte_to_line(line_offsets, span.start, total_lines),
                end_line=_byte_to_line(line_offsets, span.end, total_lines),
                language=self.language,
                metadata={"kind": span.kind.name},
            )
            chunks.append(chunk)
        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks
