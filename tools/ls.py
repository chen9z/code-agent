from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import fnmatch

from tools.base import BaseTool


class LSTool(BaseTool):
    """Tool that lists directory contents with optional ignore patterns."""

    @property
    def name(self) -> str:
        return "LS"

    @property
    def description(self) -> str:
        return """Lists files and directories in a given path. The path parameter must be an absolute path, not a relative path. You can optionally provide an array of glob patterns to ignore with the ignore parameter. You should generally prefer the Glob and Grep tools, if you know which directories to search."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute path to the directory to list (must be absolute, not relative)",
                },
                "ignore": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of glob patterns to ignore",
                },
            },
            "required": ["path"],
        }

    def execute(self, *, path: str, ignore: Iterable[str] | None = None) -> Dict[str, Any]:
        try:
            directory = Path(path)
            if not directory.is_absolute():
                raise ValueError("path must be an absolute path")

            if not directory.exists():
                raise FileNotFoundError(f"Directory does not exist: {path}")

            if not directory.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}")

            ignore_patterns = list(ignore or [])
            if not all(isinstance(pattern, str) for pattern in ignore_patterns):
                raise TypeError("ignore patterns must be strings")
            resolved = directory.resolve()

            def is_ignored(candidate: Path) -> bool:
                if not ignore_patterns:
                    return False
                relative = str(candidate.relative_to(resolved))
                name = candidate.name
                for pattern in ignore_patterns:
                    if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(name, pattern):
                        return True
                return False

            entries: List[str] = []
            directories: List[str] = []
            files: List[str] = []

            try:
                children = list(resolved.iterdir())
            except PermissionError:
                raise PermissionError(f"Permission denied listing directory: {path}")

            children.sort(key=lambda item: (item.is_file(), item.name.lower()))

            for child in children:
                if is_ignored(child):
                    continue
                display = child.name + ("/" if child.is_dir() else "")
                entries.append(display)
                if child.is_dir():
                    directories.append(display)
                else:
                    files.append(display)

            result = "\n".join(entries) if entries else "[empty directory]"

            return {
                "status": "success",
                "content": result,
                "data": {
                    "path": str(resolved),
                    "entries": entries,
                    "directories": directories,
                    "files": files,
                    "ignore": ignore_patterns,
                },
            }
        except Exception as exc:  # pragma: no cover - errors exercised via tests
            error_payload: Dict[str, Any] = {
                "error": str(exc),
                "path": path,
            }
            if ignore is not None:
                error_payload["ignore"] = list(ignore)
            error_payload["display"] = _build_error_display(str(exc))
            return {
                "status": "error",
                "content": str(exc),
                "data": error_payload,
            }


def _build_error_display(message: str) -> List[tuple[str, str]]:
    text = str(message or "").strip()
    entries: List[tuple[str, str]] = []
    if text:
        entries.append(("error", text))
    return entries
