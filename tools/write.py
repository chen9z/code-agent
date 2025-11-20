from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field

from runtime.tool_types import ToolResult
from tools.base import BaseTool


class WriteArguments(BaseModel):
    file_path: str = Field(..., description="Absolute path to the file to write.")
    content: str = Field(..., description="Content to write into the file.")

    model_config = ConfigDict(extra="forbid")


class WriteTool(BaseTool):
    """Tool that writes file contents to disk."""

    ArgumentsModel = WriteArguments

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

    def execute(self, *, file_path: str, content: str) -> ToolResult:
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
            data["display"] = success_message
            return ToolResult(status="success", content=success_message, data=data)
        except Exception as exc:  # pragma: no cover - exercised via tests
            message = str(exc)
            return ToolResult(
                status="error",
                content=message,
                data={
                    "error": message,
                    "file_path": file_path,
                    "display": _build_error_display(message),
                },
            )


def _build_error_display(message: str) -> str:
    text = str(message or "").strip()
    return text or "Unknown error"
