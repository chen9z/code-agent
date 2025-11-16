from __future__ import annotations

"""Thin wrapper around the project index adapter for indexing and search."""

import json
from pathlib import Path

from retrieval.index import create_index


def main() -> int:
    project_path = Path("~/workspace/spring-ai").expanduser()
    query_text = "查找项目中有哪些 embedding model"
    # project_path = Path("~/workspace/code-agent").expanduser()
    # query_text = "查找项目中有哪些 tools"

    if not project_path:
        raise ValueError("project_path parameter is required")

    path = project_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Project directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    try:
        project_index = create_index()
        index_info = project_index.index_project(str(path), show_progress=True)
        project_name = index_info.get("project_name", path.name)
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"Indexing failed: {exc}")
        return 1

    print("Index Result:")
    print(json.dumps(index_info, ensure_ascii=False, indent=2))

    try:
        results = project_index.search(
            project_name,
            query_text,
            target_directories=None,
        )
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"Search failed: {exc}")
        return 1

    print("\nSearch Result:")
    print(project_index.format_search_results(results))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())
