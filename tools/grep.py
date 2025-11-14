from __future__ import annotations

import json
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _build_pointer(prefix_len: int, spans: Iterable[Tuple[int, int]], line_length: int) -> Optional[str]:
    """Render a caret pointer line for highlighted spans when feasible."""
    if line_length <= 0:
        return None
    markers = [" "] * line_length
    has_marker = False
    for raw_start, raw_end in spans:
        if raw_start is None or raw_end is None:
            continue
        start = max(0, raw_start)
        end = max(start + 1, raw_end)
        if start >= line_length:
            continue
        if end > line_length:
            end = line_length
        for idx in range(start, end):
            markers[idx] = "^"
            has_marker = True
    if not has_marker:
        return None
    pointer = "".join(markers).rstrip()
    if not pointer:
        return None
    return " " * prefix_len + pointer


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
            if not entry.get("truncated") and snippet:
                pointer = _build_pointer(len(prefix), entry.get("spans", []), len(snippet))
                if pointer:
                    formatted_lines.append(pointer)
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
        return """A powerful search tool built on ripgrep

  Usage:
  - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.
  - Supports full regex syntax (e.g., "log.*Error", "function\s+\w+")
  - Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")
  - Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
  - Use Task tool for open-ended searches requiring multiple rounds
  - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\{\}` to find `interface{}` in Go code)
  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \{[\s\S]*?field`, use `multiline: true`"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "-A": {
                    "type": "number",
                    "description": "Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise."
                },
                "-B": {
                    "type": "number",
                    "description": "Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise."
                },
                "-C": {
                    "type": "number",
                    "description": "Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise."
                },
                "-i": {
                    "type": "boolean",
                    "description": "Case insensitive search (rg -i)"
                },
                "-n": {
                    "type": "boolean",
                    "description": "Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise. Defaults to true."
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. \"*.js\", \"*.{ts,tsx}\") - maps to rg --glob"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (rg PATH). Defaults to current working directory."
                },
                "type": {
                    "type": "string",
                    "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types."
                },
                "offset": {
                    "type": "number",
                    "description": "Skip first N lines/entries before applying head_limit, equivalent to \"| tail -n +N | head -N\". Works across all output modes. Defaults to 0."
                },
                "pattern": {
                    "type": "string",
                    "description": "The regular expression pattern to search for in file contents"
                },
                "multiline": {
                    "type": "boolean",
                    "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false."
                },
                "head_limit": {
                    "type": "number",
                    "description": "Limit output to first N lines/entries, equivalent to \"| head -N\". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). Defaults based on \"cap\" experiment value: 0 (unlimited), 20, or 100."
                },
                "output_mode": {
                    "enum": [
                        "content",
                        "files_with_matches",
                        "count"
                    ],
                    "type": "string",
                    "description": "Output mode: \"content\" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), \"files_with_matches\" shows file paths (supports head_limit), \"count\" shows match counts (supports head_limit). Defaults to \"files_with_matches\"."
                }
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
        exit_code: Optional[int] = None
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
