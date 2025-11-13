from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tools.base import BaseTool

MAX_LINE_LENGTH = 2000
DEFAULT_LIMIT = 2000


def format_line_with_arrow(line_number: int, content: str) -> str:
    """Return a formatted line using a fixed-width number and arrow delimiter."""
    return f"{line_number:6}→{content}"


# Tracks files that have been read along with their modification timestamps.
READ_REGISTRY: Dict[str, float] = {}


def register_read_path(path: Path) -> None:
    """Record that a file was read, storing its current modification time."""
    try:
        READ_REGISTRY[str(path)] = path.stat().st_mtime
    except FileNotFoundError:
        READ_REGISTRY.pop(str(path), None)


def get_last_read_mtime(path: Path) -> float | None:
    """Retrieve the recorded modification time for a previously read file."""
    return READ_REGISTRY.get(str(path))


def clear_read_record(path: Path) -> None:
    """Remove any stored read record for the given file."""
    READ_REGISTRY.pop(str(path), None)


class ReadTool(BaseTool):
    """Tool that reads files with optional pagination support."""

    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        return """Reads a file from the local filesystem using cat -n formatted output. Supports optional line offset and limit for large files."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["file_path"],
            "properties": {
                "limit": {
                    "type": "number",
                    "description": (
                        "Number of lines to read (default 2000). Provide to paginate very large files."
                    ),
                },
                "offset": {
                    "type": "number",
                    "description": (
                        "1-based line number to start reading from. Provide to paginate very large files."
                    ),
                },
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read.",
                },
            },
        }

    def execute(
        self,
        *,
        file_path: str,
        limit: float | int | None = None,
        offset: float | int | None = None,
    ) -> Dict[str, Any]:
        try:
            path = Path(file_path)
            if not path.is_absolute():
                raise ValueError("file_path must be an absolute path")

            if not path.exists():
                raise FileNotFoundError(f"File does not exist: {path}")

            if path.is_dir():
                raise IsADirectoryError(f"Path is a directory, not a file: {path}")

            start_line = int(offset) if offset is not None else 1
            if start_line < 1:
                raise ValueError("offset must be greater than or equal to 1")

            max_lines = int(limit) if limit is not None else DEFAULT_LIMIT
            if max_lines < 1:
                raise ValueError("limit must be greater than or equal to 1")

            resolved_path = path.resolve()
            formatted_lines: List[str] = []
            lines_read = 0
            truncated = False

            end_line = start_line + max_lines - 1
            has_more = False

            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if line_number < start_line:
                        continue
                    if line_number > end_line:
                        has_more = True
                        break

                    stripped = raw_line.rstrip("\n")
                    if len(stripped) > MAX_LINE_LENGTH:
                        stripped = stripped[:MAX_LINE_LENGTH] + "… (truncated)"
                        truncated = True

                    formatted_lines.append(format_line_with_arrow(line_number, stripped))
                    lines_read += 1

            register_read_path(resolved_path)

            result = "\n".join(formatted_lines)

            if not formatted_lines:
                result = "[empty output]"

            display_summary = f"Read {lines_read} lines" if isinstance(lines_read, int) else "Read file"

            metadata = {
                "file_path": str(resolved_path),
                "offset": start_line,
                "limit": max_lines,
                "count": lines_read,
                "has_more": has_more,
                "truncated": truncated,
                "display": [("result", display_summary)],
            }

            return {
                "status": "success",
                "content": result,
                "data": metadata,
            }
        except Exception as exc:  # pragma: no cover - exercised via tests
            message = str(exc)
            return {
                "status": "error",
                "content": message,
                "data": {
                    "error": message,
                    "file_path": file_path,
                },
            }
