from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tools.base import BaseTool


class GlobTool(BaseTool):
    """Tool that performs glob-based path searches inside a directory."""

    @property
    def name(self) -> str:
        return "Glob"

    @property
    def description(self) -> str:
        return """Fast file pattern matching tool that supports glob patterns like '**/*.js' or 'src/**/*.ts'. Returns matches sorted by modification time (newest first)."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Directory to search. Omit to use the current working directory. Must "
                        "exist and be a directory if provided."
                    ),
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

            return {
                "matches": matches,
                "search_path": str(resolved_dir),
                "content": joined,
                "count": len(matches),
            }
        except Exception as exc:
            error_payload: Dict[str, Any] = {
                "error": str(exc),
                "content": str(exc),
            }
            if path:
                error_payload["search_path"] = path
            return error_payload
