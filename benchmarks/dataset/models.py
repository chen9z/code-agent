from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class GoldenChunk:
    path: str
    start_line: int
    end_line: int
    confidence: float
    content: str
    fingerprint: str
    content_hash: str


@dataclass
class DatasetSample:
    query_id: str
    query: str
    repo_url: str
    commit: str
    golden_chunks: list[GoldenChunk]
    schema_version: str = "2025.11"
