from __future__ import annotations

import os
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tree_sitter import Parser

from integrations.tree_sitter.language_loader import (
    is_language_supported,
    load_language,
)

try:
    import pathspec  # type: ignore

    HAS_PATHSPEC = True
except Exception:
    pathspec = None  # type: ignore
    HAS_PATHSPEC = False


# Supported language detection by file extension
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


_PARSER_CACHE: Dict[str, Any] = {}


@dataclass
class Chunk:
    path: str
    content: str
    start_line: int
    end_line: int
    language: Optional[str] = None
    symbol: Optional[str] = None


def _load_gitignore_patterns(root: Path) -> List[str]:
    gi = root / ".gitignore"
    if not gi.exists():
        return []
    lines = []
    for raw in gi.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    # Always ignore VCS dir
    lines.append(".git/")
    return lines


def _make_ignore_matcher(root: Path):
    patterns = _load_gitignore_patterns(root)
    if HAS_PATHSPEC and patterns:
        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

        def matcher(p: Path) -> bool:
            rel = p.relative_to(root).as_posix()
            return spec.match_file(rel)

        return matcher

    def fallback_matcher(p: Path) -> bool:
        rel = p.relative_to(root).as_posix()
        ignored = False
        for pat in patterns:
            neg = pat.startswith("!")
            pat_eff = pat[1:] if neg else pat
            # Directory pattern
            if pat_eff.endswith("/"):
                if rel.startswith(pat_eff):
                    ignored = not neg
                continue
            if fnmatch.fnmatch(rel, pat_eff) or fnmatch.fnmatch(Path(rel).name, pat_eff):
                ignored = not neg
        # Default ignores for common dirs
        default_dirs = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "dist", "build", "target", ".pytest_cache", ".mypy_cache", ".qdrant", "storage"}
        parts = set(Path(rel).parts)
        if parts & default_dirs:
            return True
        return ignored

    return fallback_matcher


def _get_cached_parser(language: str):
    if not is_language_supported(language):
        return None
    parser = _PARSER_CACHE.get(language)
    if parser is None:
        parser = Parser()
        parser.language = load_language(language)
        _PARSER_CACHE[language] = parser
    return parser


def iter_repository_files(root: Path) -> Iterator[Path]:
    """Yield code file paths under root, respecting .gitignore when possible."""
    root = root.expanduser().resolve()
    ignore = _make_ignore_matcher(root)
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories
        to_keep = []
        for d in dirnames:
            dpath = Path(dirpath) / d
            if not ignore(dpath):
                to_keep.append(d)
        dirnames[:] = to_keep
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if ignore(fpath):
                continue
            if fpath.suffix.lower() in CODE_EXTS:
                yield fpath


def _detect_language(path: Path) -> Optional[str]:
    return EXT_LANGUAGE.get(path.suffix.lower())


def _gather_nodes(root_node, language: str):
    """Collect AST nodes representing chunk-worthy units by language."""
    # Map language to node types to include
    lang_nodes: Dict[str, List[str]] = {
        "python": ["function_definition", "class_definition"],
        "javascript": ["function_declaration", "method_definition", "class_declaration"],
        "typescript": ["function_declaration", "method_signature", "method_definition", "class_declaration"],
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
    target = set(lang_nodes.get(language, []))
    nodes = []

    def walk(n):
        if n.type in target:
            nodes.append(n)
        for c in n.children:
            walk(c)

    walk(root_node)
    return nodes


def _get_node_name(node) -> Optional[str]:
    try:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return name_node.text.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return None


def _safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("utf-8", errors="ignore")


def _compute_context_start(lines: List[str], start_line: int, base_context: int, *, max_extra: int = 6) -> int:
    """Extend context upward to include adjacent blank/comment lines."""
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


def _find_uncovered_segments(start: int, end: int, covered: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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


def chunk_code_file(
    path: Path,
    max_lines: int = 200,
    overlap: int = 0,
    pre_context: int = 2,
    *,
    chunk_size: int | None = None,
) -> List[Chunk]:
    """Chunk a single source file using tree-sitter when possible.

    Notes
    - overlap is ignored (kept for backward compatibility). Chunks are non-overlapping.
    - Each chunk will contain at most `chunk_size` (or `max_lines`) lines.
    - Falls back to simple line-based chunking when no parser is available.
    """
    size = max(1, int(chunk_size or max_lines))
    language = _detect_language(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    if total_lines == 0:
        return []

    if not language:
        # Unknown language → line-based chunks
        return _line_chunks(path, lines, size)

    try:
        parser = _get_cached_parser(language)
    except Exception:
        parser = None
    if parser is None:
        return _line_chunks(path, lines, size)

    try:
        tree = parser.parse(text.encode("utf-8"))
    except Exception:
        return _line_chunks(path, lines, size)

    root = tree.root_node
    nodes = _gather_nodes(root, language)
    if not nodes:
        return _line_chunks(path, lines, size)

    node_entries: List[Tuple[int, int, int, int, Any]] = []
    for node in nodes:
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        if end_line < start_line:
            continue
        ctx_start = _compute_context_start(lines, start_line, pre_context)
        length = end_line - start_line + 1
        node_entries.append((length, start_line, ctx_start, end_line, node))

    if not node_entries:
        return _line_chunks(path, lines, size)

    node_entries.sort(key=lambda entry: (entry[0], entry[1]))

    chunks: List[Chunk] = []
    covered: List[Tuple[int, int]] = []

    for _, start_line, ctx_start, end_line, node in node_entries:
        target_start = max(1, min(ctx_start, total_lines))
        target_end = min(total_lines, max(target_start, end_line))
        segments: List[Tuple[int, int]] = []
        has_split = False
        for seg_start, seg_end in _find_uncovered_segments(target_start, target_end, covered):
            if seg_end < seg_start:
                continue
            if seg_end - seg_start + 1 > size:
                has_split = True
                segments.extend(_split_segment(seg_start, seg_end, size))
            else:
                segments.append((seg_start, seg_end))
        if not segments:
            continue
        base_symbol = _get_node_name(node)
        use_part_labels = has_split or len(segments) > 1
        for idx, (seg_start, seg_end) in enumerate(segments, start=1):
            content = "".join(lines[seg_start - 1 : seg_end])
            if not content.strip():
                continue
            symbol = base_symbol
            if base_symbol and use_part_labels:
                symbol = f"{base_symbol} (part {idx})"
            chunks.append(
                Chunk(
                    path=str(path),
                    content=content,
                    start_line=seg_start,
                    end_line=seg_end,
                    language=language,
                    symbol=symbol,
                )
            )
            _add_coverage(covered, seg_start, seg_end)

    for seg_start, seg_end in _find_uncovered_segments(1, total_lines, covered):
        if seg_end < seg_start:
            continue
        overflow_segments: List[Tuple[int, int]]
        if seg_end - seg_start + 1 > size:
            overflow_segments = _split_segment(seg_start, seg_end, size)
        else:
            overflow_segments = [(seg_start, seg_end)]
        for piece_start, piece_end in overflow_segments:
            content = "".join(lines[piece_start - 1 : piece_end])
            if not content.strip():
                continue
            chunks.append(
                Chunk(
                    path=str(path),
                    content=content,
                    start_line=piece_start,
                    end_line=piece_end,
                    language=language,
                )
            )
            _add_coverage(covered, piece_start, piece_end)

    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks


def _line_chunks(path: Path, lines: List[str], max_lines: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    start = 1
    total = len(lines)
    while start <= total:
        end = min(total, start + max_lines - 1)
        content = "".join(lines[start - 1 : end])
        chunks.append(
            Chunk(
                path=str(path),
                content=content,
                start_line=start,
                end_line=end,
                language=_detect_language(path),
            )
        )
        start = end + 1
    return chunks
