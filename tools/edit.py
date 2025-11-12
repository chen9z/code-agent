from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from tools.base import BaseTool
from tools.read import clear_read_record, get_last_read_mtime


class EditTool(BaseTool):
    """Tool that performs exact string replacements in files."""

    @property
    def name(self) -> str:
        return "Edit"

    @property
    def description(self) -> str:
        return """Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to modify",
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to replace",
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with (must be different from old_string)",
                },
                "replace_all": {
                    "type": "boolean",
                    "default": False,
                    "description": "Replace all occurences of old_string (default false)",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    def execute(
        self,
        *,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool | None = None,
    ) -> Dict[str, Any]:
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

            replace_all_flag = bool(replace_all) if replace_all is not None else False
            if replace_all is not None and not isinstance(replace_all, bool):
                raise TypeError("replace_all must be a boolean")

            if new_string == old_string:
                raise ValueError("new_string must be different from old_string")

            file_exists = resolved.exists()

            if file_exists and resolved.is_dir():
                raise IsADirectoryError(f"Cannot edit a directory: {resolved}")

            if file_exists:
                recorded_mtime = get_last_read_mtime(resolved)
                current_mtime = resolved.stat().st_mtime
                if recorded_mtime is None or recorded_mtime != current_mtime:
                    raise PermissionError(
                        "File must be read with the Read tool before editing the existing file."
                    )

                if old_string == "":
                    raise ValueError("old_string must not be empty when editing an existing file")

                original_content = resolved.read_text(encoding="utf-8")
            else:
                if old_string != "":
                    raise FileNotFoundError(
                        "File does not exist. To create a new file, provide an empty old_string and the desired content as new_string."
                    )
                if replace_all_flag:
                    raise ValueError("replace_all cannot be true when creating a new file")
                if new_string == "":
                    raise ValueError("new_string must not be empty when creating a new file")
                original_content = ""

            if old_string == "" and file_exists:
                raise ValueError("old_string must not be empty when editing an existing file")

            if replace_all_flag:
                if old_string == "":
                    raise ValueError("old_string must not be empty when replace_all is true")
                occurrences = original_content.count(old_string)
                if occurrences == 0:
                    raise ValueError("old_string was not found in the file")
                updated_content = original_content.replace(old_string, new_string)
                replacements = occurrences
            else:
                if old_string == "":
                    if file_exists:
                        raise ValueError("old_string must not be empty when editing an existing file")
                    # Creating new file handled above; treat as direct assignment
                    updated_content = new_string
                    replacements = 1
                else:
                    occurrences = original_content.count(old_string)
                    if occurrences == 0:
                        raise ValueError("old_string was not found in the file")
                    if occurrences > 1:
                        raise ValueError("old_string matched multiple times; provide more context or use replace_all")
                    updated_content = original_content.replace(old_string, new_string, 1)
                    replacements = 1

            with resolved.open("w", encoding="utf-8") as handle:
                handle.write(updated_content)

            clear_read_record(resolved)

            return {
                "file_path": str(resolved),
                "replacements": replacements,
                "content": "ok",
            }
        except Exception as exc:  # pragma: no cover - exercised via tests
            return {
                "error": str(exc),
                "file_path": file_path,
                "content": str(exc),
            }
