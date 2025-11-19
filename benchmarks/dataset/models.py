from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    query: str
    repo_url: str
    branch: str
    commit: str
    path: Optional[str] = None  # 本地快照来源


@dataclass
class SnapshotMetadata:
    repo_url: str
    branch: str
    commit: str
    snapshot_path: str
    created_at: str
    index_built: bool = False


@dataclass
class DatasetRunResult:
    query_id: str
    success: bool
    message: Optional[str]
