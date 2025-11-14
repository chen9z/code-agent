from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tools.base import BaseTool


MAX_DISPLAY_MATCHES = 5


def _build_display(matches: List[str]) -> str:
    if not matches:
        return "No matches"

    visible_matches = matches[:MAX_DISPLAY_MATCHES]
    entries = [f"- {match}" for match in visible_matches]
    if len(matches) > MAX_DISPLAY_MATCHES:
        entries.append(f"+{len(matches) - MAX_DISPLAY_MATCHES} more")
    return "\n".join(entries)


def _error_display(message: str) -> str:
    text = str(message or "").strip()
    return text or "Unknown error"


class GlobTool(BaseTool):
    """Tool that performs glob-based path searches inside a directory."""

    @property
    def name(self) -> str:
        return "Glob"

    @property
    def description(self) -> str:
        return '''- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead
- You can call multiple tools in a single response. It is always better to speculatively perform multiple searches in parallel if they are potentially useful.'''

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": '''The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided.''',
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match entries against.",
                },
            },
        }

    def execute(self, *, pattern: str, path: str | None = None) -> Dict[str, Any]:
        try:
            search_dir = Path(path) if path else Path.cwd()
            if not search_dir.exists():
                raise FileNotFoundError(f"Search directory does not exist: {search_dir}")
            if not search_dir.is_dir():
                raise NotADirectoryError(f"Search path is not a directory: {search_dir}")

            resolved_dir = search_dir.resolve()
            matched_paths: List[Path] = []
            for match in resolved_dir.glob(pattern):
                try:
                    match.stat()
                except FileNotFoundError:
                    # File could disappear between glob and stat; skip it
                    continue
                matched_paths.append(match)

            matched_paths.sort(key=lambda p: str(p))
            matches = [str(p.relative_to(resolved_dir)) for p in matched_paths]
            joined = "\n".join(str(p) for p in matched_paths)
            content_text = joined if joined else "[no matches]"

            data = {
                "matches": matches,
                "search_path": str(resolved_dir),
                "count": len(matches),
                "display": _build_display(matches),
            }

            return {
                "status": "success",
                "content": content_text,
                "data": data,
            }
        except Exception as exc:
            message = str(exc)
            error_payload: Dict[str, Any] = {
                "error": message,
                "display": _error_display(message),
            }
            if path:
                error_payload["search_path"] = path
            return {
                "status": "error",
                "content": message,
                "data": error_payload,
            }
