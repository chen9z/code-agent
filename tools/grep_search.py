from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base import BaseTool

MAX_MATCHES = 50


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
                "explanation": {
                    "type": "string",
                    "description": (
                        "One sentence explanation as to why this tool is being used, and how it contributes to the goal."
                    ),
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
        explanation: str | None = None,
    ) -> Dict[str, Any]:
        if not query:
            return {"error": "query must be a non-empty string", "query": query}

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
            return {"error": "ripgrep (rg) is not installed or not in PATH"}

        matches: List[Dict[str, Any]] = []
        truncated = False
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
                for submatch in data.get("submatches", []):
                    match_text = submatch.get("match", {}).get("text", "")
                    start = submatch.get("start")
                    end = submatch.get("end")
                    matches.append(
                        {
                            "path": path_text,
                            "line": line_number,
                            "match": match_text,
                            "line_text": lines_text.rstrip("\n"),
                            "start": start,
                            "end": end,
                        }
                    )
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

        if truncated and exit_code not in (0, 1):
            exit_code = 0

        if exit_code not in (0, 1):
            return {
                "error": stderr_output.strip() or "ripgrep failed",
                "exit_code": exit_code,
                "query": query,
            }

        return {
            "query": query,
            "matches": matches,
            "count": len(matches),
            "truncated": truncated,
            "explanation": explanation,
            "stderr": stderr_output.strip(),
        }
