from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class SnapshotMetadata:
    repo_url: str
    branch: str
    commit: str
    snapshot_path: str
    created_at: str
    index_built: bool = False


class SnapshotManager:
    """Materialize repo snapshots under artifacts/snapshots.<br>
    Linus 风格：保持简单，依赖本地路径复制，不额外封装复杂 git 逻辑。
    """

    def __init__(self, base_dir: str | Path = "artifacts") -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.snapshots_root = self.base_dir / "snapshots"
        self.snapshots_root.mkdir(parents=True, exist_ok=True)

    def materialize(
        self,
        *,
        repo_url: str,
        branch: str = "main",
        commit: str = "working",
        source_path: Optional[str | Path] = None,
        refresh: bool = False,
    ) -> SnapshotMetadata:
        source = Path(source_path or repo_url).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"repo path not found: {source}")
        if not source.is_dir():
            raise NotADirectoryError(f"repo path is not directory: {source}")

        slug = self._slugify(source.name)
        target = self.snapshots_root / slug / commit
        target.parent.mkdir(parents=True, exist_ok=True)

        if refresh and target.exists():
            shutil.rmtree(target)

        if not target.exists():
            shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))

        return self._write_metadata(repo_url, branch, commit, target)

    def _write_metadata(
        self,
        repo_url: str,
        branch: str,
        commit: str,
        snapshot_path: Path,
    ) -> SnapshotMetadata:
        metadata = SnapshotMetadata(
            repo_url=repo_url,
            branch=branch,
            commit=commit,
            snapshot_path=str(snapshot_path.resolve()),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            index_built=False,
        )
        meta_file = Path(snapshot_path) / "snapshot_metadata.json"
        with meta_file.open("w", encoding="utf-8") as handle:
            json.dump(asdict(metadata), handle, ensure_ascii=False, indent=2)
        return metadata

    @staticmethod
    def _slugify(raw: str) -> str:
        slug = [ch for ch in raw if ch.isalnum() or ch in {"-", "_"}]
        text = "".join(slug).lower()
        return text or "repo"
