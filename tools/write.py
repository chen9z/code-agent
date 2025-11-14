from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tools.base import BaseTool


class WriteTool(BaseTool):
    """Tool that writes file contents to disk."""

    @property
    def name(self) -> str:
        return "Write"

    @property
    def description(self) -> str:
        return """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write (must be absolute, not relative)",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        }

    def execute(self, *, file_path: str, content: str) -> Dict[str, Any]:
        try:
            path = Path(file_path)
            if not path.is_absolute():
                raise ValueError("file_path must be an absolute path")

            resolved = path.resolve()
            parent = resolved.parent

            if not parent.exists():
                raise FileNotFoundError(f"Parent directory does not exist: {parent}")
            if not parent.is_dir():
                raise NotADirectoryError(f"Parent path is not a directory: {parent}")

            if resolved.exists() and resolved.is_dir():
                raise IsADirectoryError(f"Cannot write to a directory: {resolved}")

            with resolved.open("w", encoding="utf-8") as handle:
                handle.write(content)

            bytes_written = len(content.encode("utf-8"))
            success_message = f"File created successfully at: {resolved}"

            data = {
                "file_path": str(resolved),
                "bytes_written": bytes_written,
            }
            return {
                "status": "success",
                "content": success_message,
                "data": data,
            }
        except Exception as exc:  # pragma: no cover - exercised via tests
            message = str(exc)
            return {
                "status": "error",
                "content": message,
                "data": {
                    "error": message,
                    "file_path": file_path,
                    "display": _build_error_display(message),
                },
            }


def _build_error_display(message: str) -> List[tuple[str, str]]:
    text = str(message or "").strip()
    entries: List[tuple[str, str]] = []
    if text:
        entries.append(("error", text))
    return entries
