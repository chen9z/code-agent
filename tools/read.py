from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tools.base import BaseTool

MAX_LINE_LENGTH = 2000
DEFAULT_LIMIT: int | None = None


def _error_display(message: str) -> str | None:
    text = str(message or "").strip()
    return text or None


LINE_NUMBER_WIDTH = 8


def format_line_with_arrow(line_number: int, content: str) -> str:
    """Return a formatted line using a fixed-width number and arrow delimiter."""
    return f"{line_number:{LINE_NUMBER_WIDTH}}→{content}"


class ReadTool(BaseTool):
    """Tool that reads files with optional pagination support."""

    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        return '''Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 2000 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Any lines longer than 2000 characters will be truncated
- Results are returned using cat -n format, with line numbers starting at 1
- This tool can only read files, not directories. To read a directory, use an ls command via the Bash tool.
- You can call multiple tools in a single response. It is always better to speculatively read multiple potentially useful files in parallel.
- You will regularly be asked to read screenshots. If the user provides a path to a screenshot, ALWAYS use this tool to view the file at the path. This tool will work with all temporary file paths.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.'''

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["file_path"],
            "properties": {
                "limit": {
                    "type": "number",
                    "description": (
                        "The number of lines to read. Only provide if the file is too large to read at once"
                    ),
                },
                "offset": {
                    "type": "number",
                    "description": (
                        "The line number to start reading from. Only provide if the file is too large to read at once"
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

                    formatted_lines.append(format_line_with_arrow(line_number, stripped))
                    lines_read += 1

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
                "display": display_summary,
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
                    "display": _error_display(message),
                },
            }
