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
            matches_with_mtime: List[tuple[Path, float]] = []
            for match in resolved_dir.glob(pattern):
                try:
                    stat = match.stat()
                except FileNotFoundError:
                    # File could disappear between glob and stat; skip it
                    continue
                matches_with_mtime.append((match, stat.st_mtime))

            matches_with_mtime.sort(key=lambda item: item[1], reverse=True)
            matches = [str(p.relative_to(resolved_dir)) for p, _ in matches_with_mtime]
            joined = "\n".join(str(p) for p, _ in matches_with_mtime)

            return {
                "matches": matches,
                "search_path": str(resolved_dir),
                "result": joined,
                "count": len(matches),
            }
        except Exception as exc:
            error_payload: Dict[str, Any] = {
                "error": str(exc),
            }
            if path:
                error_payload["search_path"] = path
            return error_payload
