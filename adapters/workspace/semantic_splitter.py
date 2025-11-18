from __future__ import annotations

"""Tree-sitter backed semantic splitter shared by retrieval pipelines."""

import logging
from dataclasses import dataclass, field
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


@dataclass(slots=True)
class SplitterConfig:
    max_lines: int = 200
    pre_context: int = 2
    max_blank_lead: int = 6


@dataclass(slots=True)
class SemanticChunk:
    path: str
    content: str
    start_line: int
    end_line: int
    language: Optional[str]
    symbol: Optional[str] = None
    node_type: Optional[str] = None
    is_semantic: bool = False
    byte_range: Tuple[int, int] = (0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _get_parser(language: str) -> Optional[Parser]:
    if not is_language_supported(language):
        return None
    parser = _PARSER_CACHE.get(language)
    if parser is None:
        try:
            parser_obj = Parser()
            parser_obj.language = load_language(language)
            parser = parser_obj
            _PARSER_CACHE[language] = parser
        except Exception as exc:
            logger.warning("Failed to load tree-sitter language '%s': %s", language, exc)
            return None
    return parser


def _line_offsets(lines: Sequence[str]) -> List[int]:
    offsets = [0]
    total = 0
    for line in lines:
        total += len(line)
        offsets.append(total)
    return offsets


def _compute_context_start(
    lines: Sequence[str],
    start_line: int,
    base_context: int,
    *,
    max_extra: int,
) -> int:
    if start_line <= 1 or not lines:
        return 1
    idx = max(0, start_line - 1)
    ctx_start = max(0, idx - max(0, base_context))
    i = idx - 1
    extra = 0
    while i >= 0 and extra < max_extra:
        stripped = lines[i].lstrip()
        if not stripped:
            ctx_start = i
        elif stripped.startswith(("#", "//", "/*", "*", "--")) or stripped.startswith('"""') or stripped.startswith("'''"):
            ctx_start = i
        else:
            break
        i -= 1
        extra += 1
    return ctx_start + 1


def _find_uncovered_segments(
    start: int,
    end: int,
    covered: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    if start > end:
        return []
    segments: List[Tuple[int, int]] = []
    cur = start
    for c_start, c_end in covered:
        if c_end < cur:
            continue
        if c_start > end:
            break
        if c_start > cur:
            segments.append((cur, min(end, c_start - 1)))
        cur = max(cur, c_end + 1)
        if cur > end:
            break
    if cur <= end:
        segments.append((cur, end))
    return segments


def _add_coverage(covered: List[Tuple[int, int]], start: int, end: int) -> None:
    if start > end:
        return
    if not covered:
        covered.append((start, end))
        return
    new_ranges: List[Tuple[int, int]] = []
    inserted = False
    for c_start, c_end in covered:
        if c_end + 1 < start:
            new_ranges.append((c_start, c_end))
            continue
        if end + 1 < c_start:
            if not inserted:
                new_ranges.append((start, end))
                inserted = True
            new_ranges.append((c_start, c_end))
            continue
        start = min(start, c_start)
        end = max(end, c_end)
    if not inserted:
        new_ranges.append((start, end))
    covered[:] = new_ranges


def _split_segment(start: int, end: int, limit: int) -> List[Tuple[int, int]]:
    if start > end:
        return []
    if limit <= 0:
        return [(start, end)]
    pieces: List[Tuple[int, int]] = []
    s = start
    while s <= end:
        e = min(end, s + limit - 1)
        pieces.append((s, e))
        s = e + 1
    return pieces


def _get_node_name(node) -> Optional[str]:
    try:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return name_node.text.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return None


def _extract_docstring(code_snippet: str, language: str) -> Optional[str]:
    if language != "python":
        return None
    lines = code_snippet.splitlines()
    if len(lines) < 2:
        return None
    first_body_idx = 1
    while first_body_idx < len(lines) and not lines[first_body_idx].strip():
        first_body_idx += 1
    if first_body_idx >= len(lines):
        return None
    candidate = lines[first_body_idx].lstrip()
    if not candidate.startswith(('"""', "'''")):
        return None
    delimiter = candidate[:3]
    content = candidate[3:]
    doc_lines: List[str] = []
    if content.endswith(delimiter) and len(content) > 3:
        doc_lines.append(content[:-3])
        return "\n".join(doc_lines).strip() or None
    if content:
        doc_lines.append(content)
    for line in lines[first_body_idx + 1 :]:
        stripped = line.strip()
        if stripped.endswith(delimiter):
            doc_lines.append(stripped[:-3])
            break
        doc_lines.append(stripped)
    doc = "\n".join(doc_lines).strip()
    return doc or None


def _node_types(language: str) -> Sequence[str]:
    return _NODE_TYPES.get(language, [])


class SemanticSplitter:
    """Split source text into semantic/code-aware chunks."""

    def __init__(self, language: str, config: Optional[SplitterConfig] = None) -> None:
        self.language = language
        self.config = config or SplitterConfig()

    def split(self, path: str, text: str) -> List[SemanticChunk]:
        text = text or ""
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        if total_lines == 0:
            return []
        parser = _get_parser(self.language)
        if parser is None:
            return self._line_chunks(path, lines)
        try:
            tree = parser.parse(text.encode("utf-8"))
        except Exception as exc:
            logger.warning("tree-sitter parse failure for %s (%s)", path, exc)
            return self._line_chunks(path, lines)
        root = tree.root_node
        if root is None:
            return self._line_chunks(path, lines)
        node_candidates = self._gather_nodes(root)
        if not node_candidates:
            return self._line_chunks(path, lines)

        chunks: List[SemanticChunk] = []
        covered: List[Tuple[int, int]] = []
        offsets = _line_offsets(lines)

        for node in node_candidates:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            if end_line < start_line:
                continue
            ctx_start = _compute_context_start(
                lines,
                start_line,
                self.config.pre_context,
                max_extra=self.config.max_blank_lead,
            )
            target_start = max(1, min(ctx_start, total_lines))
            target_end = min(total_lines, max(target_start, end_line))
            segments: List[Tuple[int, int]] = []
            has_split = False
            for seg_start, seg_end in _find_uncovered_segments(target_start, target_end, covered):
                if seg_end < seg_start:
                    continue
                if seg_end - seg_start + 1 > self.config.max_lines:
                    has_split = True
                    segments.extend(_split_segment(seg_start, seg_end, self.config.max_lines))
                else:
                    segments.append((seg_start, seg_end))
            if not segments:
                continue
            base_symbol = _get_node_name(node)
            use_part_labels = has_split or len(segments) > 1
            docstring = None
            for idx, (seg_start, seg_end) in enumerate(segments, start=1):
                content = "".join(lines[seg_start - 1 : seg_end])
                if not content.strip():
                    continue
                symbol_name = base_symbol
                if base_symbol and use_part_labels:
                    symbol_name = f"{base_symbol} (part {idx})"
                if docstring is None and base_symbol:
                    docstring = _extract_docstring(content, self.language)
                chunk = SemanticChunk(
                    path=path,
                    content=content,
                    start_line=seg_start,
                    end_line=seg_end,
                    language=self.language,
                    symbol=symbol_name,
                    node_type=node.type,
                    is_semantic=bool(base_symbol),
                    byte_range=(offsets[seg_start - 1], offsets[seg_end]),
                    metadata={
                        "docstring": docstring,
                        "node_type": node.type,
                    },
                )
                chunks.append(chunk)
                _add_coverage(covered, seg_start, seg_end)

        for seg_start, seg_end in _find_uncovered_segments(1, total_lines, covered):
            pieces = _split_segment(seg_start, seg_end, self.config.max_lines)
            for piece_start, piece_end in pieces:
                content = "".join(lines[piece_start - 1 : piece_end])
                if not content.strip():
                    continue
                chunk = SemanticChunk(
                    path=path,
                    content=content,
                    start_line=piece_start,
                    end_line=piece_end,
                    language=self.language,
                    symbol=None,
                    node_type=None,
                    is_semantic=False,
                    byte_range=(offsets[piece_start - 1], offsets[piece_end]),
                )
                chunks.append(chunk)
                _add_coverage(covered, piece_start, piece_end)

        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks

    def _line_chunks(self, path: str, lines: Sequence[str]) -> List[SemanticChunk]:
        chunks: List[SemanticChunk] = []
        size = max(1, self.config.max_lines)
        start = 1
        total = len(lines)
        offsets = _line_offsets(lines)
        while start <= total:
            end = min(total, start + size - 1)
            content = "".join(lines[start - 1 : end])
            chunks.append(
                SemanticChunk(
                    path=path,
                    content=content,
                    start_line=start,
                    end_line=end,
                    language=self.language,
                    byte_range=(offsets[start - 1], offsets[end]),
                )
            )
            start = end + 1
        return chunks

    def _gather_nodes(self, root_node) -> List[Any]:
        queue: List[Any] = []
        targets = set(_node_types(self.language))
        if not targets:
            return []

        def walk(node):
            if node.type in targets:
                queue.append(node)
            for child in node.children:
                walk(child)

        walk(root_node)
        return queue
