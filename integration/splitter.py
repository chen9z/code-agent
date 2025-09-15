from __future__ import annotations

import os
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Optional

from tree_sitter_languages import get_parser

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
    size = int(chunk_size or max_lines)
    language = _detect_language(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines(keepends=True)

    if not language:
        # Unknown language → line-based chunks
        return _line_chunks(path, lines, size)

    try:
        parser = get_parser(language)
    except Exception:
        return _line_chunks(path, lines, size)

    try:
        tree = parser.parse(text.encode("utf-8"))
    except Exception:
        return _line_chunks(path, lines, size)

    root = tree.root_node
    nodes = _gather_nodes(root, language)
    chunks: List[Chunk] = []

    if not nodes:
        return _line_chunks(path, lines, size)

    for node in nodes:
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        # include a bit of pre-context
        start_line_with_ctx = max(1, start_line - pre_context)

        if end_line - start_line_with_ctx + 1 <= size:
            content = "".join(lines[start_line_with_ctx - 1 : end_line])
            chunks.append(
                Chunk(
                    path=str(path),
                    content=content,
                    start_line=start_line_with_ctx,
                    end_line=end_line,
                    language=language,
                    symbol=_get_node_name(node),
                )
            )
        else:
            # Split large definitions by contiguous segments (no overlap)
            s = start_line_with_ctx
            while s <= end_line:
                e = min(end_line, s + size - 1)
                content = "".join(lines[s - 1 : e])
                sym = _get_node_name(node)
                if sym:
                    # Compute part index based on contiguous segmentation
                    part_index = ((s - start_line_with_ctx) // max(1, size)) + 1
                    sym = f"{sym} (part {part_index})"
                chunks.append(
                    Chunk(
                        path=str(path),
                        content=content,
                        start_line=s,
                        end_line=e,
                        language=language,
                        symbol=sym,
                    )
                )
                if e == end_line:
                    break
                s = e + 1

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
