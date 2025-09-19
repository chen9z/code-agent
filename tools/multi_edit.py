from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tools.base import BaseTool
from tools.read import clear_read_record, get_last_read_mtime


class MultiEditTool(BaseTool):
    """Tool that performs multiple sequential string replacements within a file."""

    @property
    def name(self) -> str:
        return "MultiEdit"

    @property
    def description(self) -> str:
        return """This is a tool for making multiple edits to a single file in one operation. It is built on top of the Edit tool and allows you to perform multiple find-and-replace operations efficiently. Prefer this tool over the Edit tool when you need to make multiple edits to the same file.

Before using this tool:

1. Use the Read tool to understand the file's contents and context
2. Verify the directory path is correct

To make multiple file edits, provide the following:
1. file_path: The absolute path to the file to modify (must be absolute, not relative)
2. edits: An array of edit operations to perform, where each edit contains:
   - old_string: The text to replace (must match the file contents exactly, including all whitespace and indentation)
   - new_string: The edited text to replace the old_string
   - replace_all: Replace all occurences of old_string. This parameter is optional and defaults to false.

IMPORTANT:
- All edits are applied in sequence, in the order they are provided
- Each edit operates on the result of the previous edit
- All edits must be valid for the operation to succeed - if any edit fails, none will be applied
- This tool is ideal when you need to make several changes to different parts of the same file
- For Jupyter notebooks (.ipynb files), use the NotebookEdit instead

CRITICAL REQUIREMENTS:
1. All edits follow the same requirements as the single Edit tool
2. The edits are atomic - either all succeed or none are applied
3. Plan your edits carefully to avoid conflicts between sequential operations

WARNING:
- The tool will fail if edits.old_string doesn't match the file contents exactly (including whitespace)
- The tool will fail if edits.old_string and edits.new_string are the same
- Since edits are applied in sequence, ensure that earlier edits don't affect the text that later edits are trying to find

When making edits:
- Ensure all edits result in idiomatic, correct code
- Do not leave the code in a broken state
- Always use absolute file paths (starting with /)
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- Use replace_all for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.

If you want to create a new file, use:
- A new file path, including dir name if needed
- First edit: empty old_string and the new file's contents as new_string
- Subsequent edits: normal edit operations on the created content"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to modify",
                },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {
                                "type": "string",
                                "description": "The text to replace",
                            },
                            "new_string": {
                                "type": "string",
                                "description": "The text to replace it with",
                            },
                            "replace_all": {
                                "type": "boolean",
                                "default": False,
                                "description": "Replace all occurences of old_string (default false).",
                            },
                        },
                        "required": ["old_string", "new_string"],
                    },
                    "minItems": 1,
                    "description": "Array of edit operations to perform sequentially on the file",
                },
            },
            "required": ["file_path", "edits"],
        }

    def execute(self, *, file_path: str, edits: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            if not edits:
                raise ValueError("edits must contain at least one operation")

            path = Path(file_path)
            if not path.is_absolute():
                raise ValueError("file_path must be an absolute path")

            resolved = path.resolve()
            parent = resolved.parent

            if not parent.exists():
                raise FileNotFoundError(f"Parent directory does not exist: {parent}")
            if not parent.is_dir():
                raise NotADirectoryError(f"Parent path is not a directory: {parent}")

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
                working_content = resolved.read_text(encoding="utf-8")
            else:
                working_content = ""

            replacements: List[int] = []

            for index, edit in enumerate(edits):
                if not isinstance(edit, dict):
                    raise TypeError("Each edit must be an object with old_string and new_string")
                old_string = edit.get("old_string")
                new_string = edit.get("new_string")
                replace_all_flag = bool(edit.get("replace_all", False))

                if edit.get("replace_all") is not None and not isinstance(edit.get("replace_all"), bool):
                    raise TypeError("replace_all must be a boolean when provided")

                if old_string is None or new_string is None:
                    raise ValueError("Each edit must include old_string and new_string")

                if new_string == old_string:
                    raise ValueError("new_string must be different from old_string in each edit")

                if not file_exists and index == 0:
                    # New file creation path
                    if old_string != "":
                        raise FileNotFoundError(
                            "File does not exist. To create a new file, the first edit must use an empty old_string."
                        )
                    if replace_all_flag:
                        raise ValueError("replace_all cannot be true when creating a new file")
                    if new_string == "":
                        raise ValueError("new_string must not be empty when creating a new file")
                    working_content = new_string
                    replacements.append(1)
                    file_exists = True  # Subsequent edits operate on created content
                    continue

                if old_string == "":
                    raise ValueError("old_string must not be empty after the initial new-file creation edit")

                occurrences = working_content.count(old_string)
                if replace_all_flag:
                    if occurrences == 0:
                        raise ValueError("old_string was not found in the file")
                    working_content = working_content.replace(old_string, new_string)
                    replacements.append(occurrences)
                else:
                    if occurrences == 0:
                        raise ValueError("old_string was not found in the file")
                    if occurrences > 1:
                        raise ValueError(
                            "old_string matched multiple times; provide more context or use replace_all in the relevant edit"
                        )
                    working_content = working_content.replace(old_string, new_string, 1)
                    replacements.append(1)

            with resolved.open("w", encoding="utf-8") as handle:
                handle.write(working_content)

            clear_read_record(resolved)

            return {
                "file_path": str(resolved),
                "applied_edits": len(edits),
                "replacements": replacements,
                "result": "ok",
            }
        except Exception as exc:  # pragma: no cover - exercised via tests
            return {
                "error": str(exc),
                "file_path": file_path,
            }
