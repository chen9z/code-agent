from __future__ import annotations

import os
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from adapters.workspace.semantic_splitter import (
    CODE_EXTS,
    EXT_LANGUAGE,
    Document,
    SemanticSplitter,
)

try:
    import pathspec  # type: ignore

    HAS_PATHSPEC = True
except Exception:
    pathspec = None  # type: ignore
    HAS_PATHSPEC = False


@dataclass
class Chunk:
    path: str
    content: str
    start_line: int
    end_line: int
    language: Optional[str] = None
    metadata: Optional[dict] = None


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
        default_dirs = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "dist", "build", "target",
                        ".pytest_cache", ".mypy_cache", ".qdrant", "storage"}
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


def chunk_code_file(
        path: Path,
        chunk_size: int = 2048
) -> List[Chunk]:
    """Chunk a single source file using the shared SemanticSplitter."""

    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text:
        return []

    language = EXT_LANGUAGE.get(path.suffix.lower())
    if not language:
        return _line_chunks_from_text(str(path), text, chunk_size, language=None)

    splitter = SemanticSplitter(language, chunk_size)
    semantic_chunks = splitter.split(str(path), text)
    if not semantic_chunks:
        return _line_chunks_from_text(str(path), text, chunk_size, language=language)
    return [_convert_chunk(chunk) for chunk in semantic_chunks]


def _convert_chunk(chunk: Document) -> Chunk:
    return Chunk(
        path=chunk.path,
        content=chunk.content,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        language=chunk.language,
        metadata=chunk.metadata or {},
    )


def _line_chunks_from_text(
        path: str,
        text: str,
        max_lines: int,
        *,
        language: Optional[str],
) -> List[Chunk]:
    lines = text.splitlines(keepends=True)
    chunks: List[Chunk] = []
    start = 1
    total = len(lines)
    while start <= total:
        end = min(total, start + max_lines - 1)
        content = "".join(lines[start - 1: end])
        chunks.append(
            Chunk(
                path=path,
                content=content,
                start_line=start,
                end_line=end,
                language=language,
            )
        )
        start = end + 1
    return chunks
