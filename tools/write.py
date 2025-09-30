from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from tools.base import BaseTool
from tools.read import clear_read_record, get_last_read_mtime


class WriteTool(BaseTool):
    """Tool that writes file contents while enforcing prior reads."""

    @property
    def name(self) -> str:
        return "Write"

    @property
    def description(self) -> str:
        return """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.
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

            if resolved.exists():
                recorded_mtime = get_last_read_mtime(resolved)
                current_mtime = resolved.stat().st_mtime
                if recorded_mtime is None or recorded_mtime != current_mtime:
                    raise PermissionError(
                        "File must be read with the Read tool before writing to an existing file."
                    )

            with resolved.open("w", encoding="utf-8") as handle:
                handle.write(content)

            clear_read_record(resolved)

            bytes_written = len(content.encode("utf-8"))
            success_message = f"File created successfully at: {resolved}"

            return {
                "file_path": str(resolved),
                "bytes_written": bytes_written,
                "result": success_message,
            }
        except Exception as exc:  # pragma: no cover - exercised via tests
            return {
                "error": str(exc),
                "file_path": file_path,
            }
