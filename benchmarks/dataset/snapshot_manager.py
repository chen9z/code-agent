from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


@dataclass
class SnapshotMetadata:
    repo_url: str
    branch: str
    commit: str
    snapshot_path: str
    created_at: str
    index_built: bool = False


class SnapshotManager:
    """Materialize repo snapshots under storage/dataset/snapshots.<br>
    Linus 风格：保持简单，依赖本地路径复制，不额外封装复杂 git 逻辑。
    """

    def __init__(self, base_dir: str | Path = "storage/dataset") -> None:
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
        source = Path(source_path).expanduser().resolve() if source_path else None
        slug_source = source.name if source else self._repo_basename(repo_url)
        slug = self._slugify(slug_source)
        target = self.snapshots_root / slug / commit
        target.parent.mkdir(parents=True, exist_ok=True)

        if refresh and target.exists():
            shutil.rmtree(target)

        if not target.exists():
            if source is not None:
                self._copy_from_local(source=source, target=target)
            else:
                self._clone_repo(repo_url=repo_url, branch=branch, commit=commit, target=target)

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

    @staticmethod
    def _repo_basename(repo_url: str) -> str:
        if not repo_url:
            return "repo"
        parsed = urlparse(repo_url)
        candidate = parsed.path if parsed.scheme else repo_url
        name = Path(candidate.rstrip("/" )).name
        if name.endswith(".git"):
            name = name[:-4]
        return name or "repo"

    def _copy_from_local(self, *, source: Path, target: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(f"repo path not found: {source}")
        if not source.is_dir():
            raise NotADirectoryError(f"repo path is not directory: {source}")
        shutil.copytree(
            source,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", "artifacts", "storage"),
        )

    def _clone_repo(self, *, repo_url: str, branch: str, commit: str, target: Path) -> None:
        tmp_dir = target.parent / f"{target.name}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        clone_cmd = ["git", "clone"]
        if branch:
            clone_cmd.extend(["--branch", branch])
        clone_cmd.extend([repo_url, str(tmp_dir)])
        self._run_git(clone_cmd, error_prefix="git clone failed")
        checkout_ref = commit if commit and commit != "working" else branch
        if checkout_ref:
            self._run_git([
                "git",
                "-C",
                str(tmp_dir),
                "checkout",
                checkout_ref,
            ], error_prefix="git checkout failed")
        git_dir = tmp_dir / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)
        shutil.move(str(tmp_dir), str(target))

    def _run_git(self, args: list[str], *, error_prefix: str) -> None:
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"{error_prefix}: {' '.join(args)}\nstderr: {result.stderr.strip()}"
            )
