from __future__ import annotations

"""Thin wrapper around the repository adapter for indexing and search."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from integrations.repository import create_repository
_DEFAULT_LIMIT = 5


def index_project(project_path: str) -> Dict[str, Any]:
    """Index a repository path using the shared repository adapter."""

    if not project_path:
        raise ValueError("project_path parameter is required")

    path = Path(project_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Project directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    repository = create_repository()
    index_info = repository.index_project(str(path), show_progress=True)

    project_name = index_info.get("project_name", path.name)
    return {
        "status": "success",
        "project_name": project_name,
        "message": f"Successfully indexed project: {project_name}",
        "project_path": str(path),
        "action": "index",
        "chunk_count": index_info.get("chunk_count"),
        "file_count": index_info.get("file_count"),
        "chunk_size": index_info.get("chunk_size"),
    }


def search_project(
    project_name: str,
    query: str,
    *,
    limit: int = _DEFAULT_LIMIT,
    target_directories: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Execute semantic search over an indexed project via the repository adapter."""

    if not project_name:
        raise ValueError("project_name parameter is required")
    if not query:
        raise ValueError("query parameter is required")

    repository = create_repository()

    try:
        results = repository.search(
            project_name,
            query,
            max(1, int(limit)),
            target_directories=target_directories,
        )
    except Exception as exc:  # pragma: no cover - repository failures
        return {
            "status": "error",
            "message": f"Search failed: {exc}",
            "project_name": project_name,
            "query": query,
            "matches": [],
            "total_results": 0,
            "action": "search",
            "target_directories": list(target_directories or []),
        }

    matches: List[Dict[str, Any]] = []
    for doc in results:
        matches.append(
            {
                "file": doc.path,
                "chunk_id": getattr(doc, "chunk_id", None),
                "content": doc.content,
                "score": getattr(doc, "score", 0.0),
                "start_line": getattr(doc, "start_line", 0),
                "end_line": getattr(doc, "end_line", 0),
            }
        )

    return {
        "status": "success",
        "project_name": project_name,
        "query": query,
        "matches": matches,
        "total_results": len(matches),
        "action": "search",
        "target_directories": list(target_directories or []),
    }

def main() -> int:
    project_path = Path("~/workspace/spring-ai").expanduser()
    query_text = "查找项目中有哪些 embedding model"

    try:
        index_result = index_project(str(project_path))
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"Indexing failed: {exc}")
        return 1

    print("Index Result:")
    print(json.dumps(index_result, ensure_ascii=False, indent=2))

    project_name = index_result.get("project_name", project_path.name)

    try:
        search_result = search_project(project_name, query_text, limit=8)
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"Search failed: {exc}")
        return 1

    print("\nSearch Result:")
    print(json.dumps(search_result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())
