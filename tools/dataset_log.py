from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from tools.base import BaseTool


@dataclass(frozen=True)
class DatasetQueryContext:
    """Immutable metadata injected by the orchestrator for the active query."""

    query_id: str
    query: str
    repo_url: str
    branch: str
    commit: str
    snapshot_path: Path


class DatasetLogTool(BaseTool):
    """Tool that validates golden chunks and persists raw_samples JSONL."""

    def __init__(
        self,
        *,
        context: DatasetQueryContext,
        artifacts_root: str | Path | None = None,
        run_name: Optional[str] = None,
    ) -> None:
        self.context = context
        self.snapshot_path = context.snapshot_path.expanduser().resolve()
        self.artifacts_root = Path(artifacts_root or "storage/dataset").expanduser().resolve()
        self.run_name = run_name or datetime.now(timezone.utc).strftime("%Y%m%d")
        self.run_dir = self.artifacts_root / self.run_name
        self.raw_samples_dir = self.run_dir / "raw_samples"
        self.raw_samples_dir.mkdir(parents=True, exist_ok=True)
        self._seen_ranges: Set[Tuple[str, int, int]] = set()

    # ------------------------------------------------------------------ metadata
    @property
    def name(self) -> str:
        # OpenAI 工具名只允许字母/数字/下划线/减号，保持语义一致但使用下划线形式
        return "dataset_log_write_chunk"

    @property
    def description(self) -> str:
        return "Validate a golden chunk against the prepared snapshot and persist raw_samples."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["path", "start_line", "end_line", "confidence"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file relative to the snapshot root.",
                },
                "start_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based starting line of the snippet (inclusive).",
                },
                "end_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based ending line of the snippet (inclusive).",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score between 0 and 1.",
                },
            },
        }

    # ------------------------------------------------------------------ execution
    def execute(
        self,
        *,
        path: str,
        start_line: int,
        end_line: int,
        confidence: float,
    ) -> Dict[str, Any]:
        try:
            normalized = self._normalize_path(path)
            absolute = (self.snapshot_path / normalized).resolve()
            self._assert_within_snapshot(absolute)
            if not absolute.exists() or not absolute.is_file():
                raise FileNotFoundError(f"File does not exist in snapshot: {normalized}")

            start = int(start_line)
            end = int(end_line)
            if start < 1 or end < start:
                raise ValueError("start_line and end_line must describe a valid range")

            snippet, total_lines = self._read_snippet(absolute, start, end)
            if end > total_lines:
                raise ValueError(
                    f"end_line {end} exceeds file length {total_lines} for {normalized}"
                )

            conf_value = float(confidence)
            if conf_value < 0 or conf_value > 1:
                raise ValueError("confidence must be between 0 and 1")

            range_key = self._range_key(str(normalized.as_posix()), start, end)
            record = self._build_record(
                normalized,
                start,
                end,
                conf_value,
                snippet,
            )
            if range_key in self._seen_ranges:
                raise ValueError("chunk already recorded for this query")
            self._seen_ranges.add(range_key)
            self._persist_record(record)
            display = (
                f"validated {record['chunk']['path']}:{record['chunk']['start_line']}-"
                f"{record['chunk']['end_line']}"
            )
            return {
                "status": "success",
                "content": display,
                "data": {
                    "display": display,
                    "chunk": record["chunk"],
                },
            }
        except Exception as exc:  # pragma: no cover - exercised via tests
            message = str(exc)
            return {
                "status": "error",
                "content": message,
                "data": {
                    "display": message,
                },
            }

    # ------------------------------------------------------------------ helpers
    def _normalize_path(self, raw: str) -> Path:
        candidate = Path(raw)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(self.snapshot_path)
            except ValueError as exc:
                raise ValueError("path must reside within the snapshot root") from exc
        return candidate

    def _assert_within_snapshot(self, absolute: Path) -> None:
        try:
            absolute.relative_to(self.snapshot_path)
        except ValueError as exc:
            raise ValueError("Resolved path escapes the snapshot root") from exc

    def _read_snippet(self, file_path: Path, start: int, end: int) -> tuple[str, int]:
        lines: list[str] = []
        total = 0
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for idx, line in enumerate(handle, start=1):
                total = idx
                if start <= idx <= end:
                    lines.append(line.rstrip("\n"))
        snippet = "\n".join(lines)
        return snippet, total

    def _build_record(
        self,
        relative_path: Path,
        start: int,
        end: int,
        confidence: float,
        snippet: str,
    ) -> Dict[str, Any]:
        normalized_path = str(relative_path.as_posix())
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        chunk = {
            "path": normalized_path,
            "start_line": start,
            "end_line": end,
            "confidence": round(confidence, 4),
            "content": snippet,
        }
        record = {
            "timestamp": timestamp,
            "query_id": self.context.query_id,
            "query": self.context.query,
            "repo_url": self.context.repo_url,
            "branch": self.context.branch,
            "commit": self.context.commit,
            "snapshot": {
                "root": str(self.snapshot_path),
                "path": chunk["path"],
            },
            "chunk": chunk,
        }
        return record

    def _persist_record(self, record: Dict[str, Any]) -> None:
        raw_file = self.raw_samples_dir / f"{self.context.query_id}.jsonl"
        with raw_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    def _range_key(self, path: str, start: int, end: int) -> Tuple[str, int, int]:
        return (path, start, end)
