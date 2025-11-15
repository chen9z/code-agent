from __future__ import annotations

import json
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.base import BaseTool

MAX_MATCHES = 50
MAX_CONTEXT_CHARS = 400
MAX_DISPLAY_MATCHES = 5


def _sanitize_line_text(text: str) -> Tuple[str, bool]:
    """Normalize tabs, strip newline, and return whether truncation occurred."""
    normalized = text.rstrip("\n").replace("\t", "    ")
    if len(normalized) <= MAX_CONTEXT_CHARS:
        return normalized, False
    clipped = normalized[:MAX_CONTEXT_CHARS] + "… (truncated)"
    return clipped, True


def _format_grouped_matches(
    grouped: OrderedDict[str, OrderedDict[int | None, Dict[str, Any]]]
) -> str:
    if not grouped:
        return "[no matches]"

    formatted_lines: List[str] = []
    for path, line_map in grouped.items():
        formatted_lines.append(str(path))
        for line_number, entry in line_map.items():
            line_label = f"{line_number:6}" if isinstance(line_number, int) else " " * 6
            prefix = f"  {line_label}→"
            snippet = entry.get("text", "")
            formatted_lines.append(f"{prefix}{snippet}")
        formatted_lines.append("")

    if formatted_lines and formatted_lines[-1] == "":
        formatted_lines.pop()

    return "\n".join(formatted_lines) if formatted_lines else "[no matches]"


def _success_response(
    *,
    content: str,
    data: Dict[str, Any],
    display: Optional[str] = None,
) -> Dict[str, Any]:
    payload_data = dict(data)
    if display:
        payload_data["display"] = display
    return {
        "status": "success",
        "content": content,
        "data": payload_data,
    }


def _error_response(*, message: str, data: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(data)
    payload["display"] = _build_error_display(message)
    payload.setdefault("error", message)
    return {
        "status": "error",
        "content": message,
        "data": payload,
    }


def _build_display_matches(matches: List[Dict[str, Any]]) -> str:
    if not matches:
        return "No matches"

    preview = matches[:MAX_DISPLAY_MATCHES]
    lines: List[str] = []
    for match in preview:
        if not isinstance(match, dict):
            continue
        path = match.get("path") or "[unknown path]"
        line_no = match.get("line")
        location = f"{path}:{line_no}" if isinstance(line_no, int) else str(path)
        snippet = (match.get("line_text") or "").strip()
        combined = f"{location} {snippet}".strip()
        lines.append(combined or location)

    if len(matches) > len(preview):
        lines.append(f"+{len(matches) - len(preview)} more matches")

    return "\n".join(lines)


def _build_error_display(message: str) -> str:
    text = str(message or "").strip()
    return text or "Unknown error"


class GrepSearchTool(BaseTool):
    """Tool that performs regex searches using ripgrep."""

    @property
    def name(self) -> str:
        return "Grep"

    @property
    def description(self) -> str:
        return r"""### Instructions:
This is best for finding exact text matches or regex patterns.
This is preferred over semantic search when we know the exact symbol/function name/etc. to search in some set of directories/file types.

Use this tool to run fast, exact regex searches over text files using the `ripgrep` engine.
To avoid overwhelming output, the results are capped at 50 matches.
Use the include or exclude patterns to filter the search scope by file type or specific paths.

- Always escape special regex characters: ( ) [ ] { } + * ? ^ $ | . \
- Use `\` to escape any of these characters when they appear in your search string.
- Do NOT perform fuzzy or semantic matches.
- Return only a valid regex pattern string.

### Examples:
| Literal               | Regex Pattern            |
|-----------------------|--------------------------|
| function(             | function\(              |
| value[index]          | value\[index\]         |
| file.txt               | file\.txt                |
| user|admin            | user\|admin             |
| path\to\file         | path\\to\\file        |
| hello world           | hello world              |
| foo\(bar\)          | foo\\(bar\\)         |"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case sensitive",
                },
                "include_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to include (e.g. '*.ts' for TypeScript files)",
                },
                "exclude_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to exclude",
                },
            },
            "required": ["query"],
        }

    def execute(
        self,
        *,
        query: str,
        case_sensitive: bool | None = None,
        include_pattern: str | None = None,
        exclude_pattern: str | None = None,
    ) -> Dict[str, Any]:
        if not query:
            message = "query must be a non-empty string"
            return _error_response(message=message, data={"query": query})

        command: List[str] = [
            "rg",
            "--json",
            "--with-filename",
            "--line-number",
            "--column",
            "--color",
            "never",
        ]

        if case_sensitive is True:
            command.append("--case-sensitive")
        elif case_sensitive is False:
            command.append("--ignore-case")

        if include_pattern:
            command.append(f"--glob={include_pattern}")
        if exclude_pattern:
            command.append(f"--glob=!{exclude_pattern}")

        command.append(query)
        command.append(str(Path.cwd()))

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            message = "ripgrep (rg) is not installed or not in PATH"
            return _error_response(message=message, data={"query": query})

        matches: List[Dict[str, Any]] = []
        truncated = False
        snippet_truncated = False
        grouped: OrderedDict[str, OrderedDict[int | None, Dict[str, Any]]] = OrderedDict()
        stderr_output = ""
        try:
            assert process.stdout is not None
            for line in process.stdout:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if message.get("type") != "match":
                    continue

                data = message.get("data", {})
                path_text = data.get("path", {}).get("text")
                line_number = data.get("line_number")
                lines_text = data.get("lines", {}).get("text", "")
                sanitized_line, line_was_truncated = _sanitize_line_text(lines_text)
                snippet_truncated = snippet_truncated or line_was_truncated

                for submatch in data.get("submatches", []):
                    match_text = submatch.get("match", {}).get("text", "")
                    start = submatch.get("start")
                    end = submatch.get("end")
                    matches.append(
                        {
                            "path": path_text,
                            "line": line_number,
                            "match": match_text,
                            "line_text": sanitized_line,
                            "start": start,
                            "end": end,
                        }
                    )
                    clean_path = path_text or "[unknown path]"
                    line_key = line_number if isinstance(line_number, int) else None
                    file_group = grouped.setdefault(clean_path, OrderedDict())
                    entry = file_group.get(line_key)
                    if entry is None:
                        entry = {
                            "text": sanitized_line,
                            "spans": [],
                            "truncated": line_was_truncated,
                        }
                        file_group[line_key] = entry
                    entry["spans"].append((start, end))
                    entry["truncated"] = entry["truncated"] or line_was_truncated
                    if len(matches) >= MAX_MATCHES:
                        truncated = True
                        process.terminate()
                        break
                if truncated:
                    break
            process.wait()
        finally:
            if process.stderr:
                try:
                    stderr_output = process.stderr.read()
                except ValueError:
                    stderr_output = ""
            exit_code = process.returncode

        if (truncated or snippet_truncated) and exit_code not in (0, 1):
            exit_code = 0

        if exit_code not in (0, 1):
            message = stderr_output.strip() or "ripgrep failed"
            return _error_response(
                message=message,
                data={"query": query, "exit_code": exit_code},
            )

        formatted_output = _format_grouped_matches(grouped)
        display_text = _build_display_matches(matches)

        return _success_response(
            content=formatted_output,
            data={
                "query": query,
                "matches": matches,
            },
            display=display_text,
        )
