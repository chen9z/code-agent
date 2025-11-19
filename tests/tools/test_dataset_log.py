from __future__ import annotations

from pathlib import Path

import pytest

from tools.dataset_log import DatasetLogTool, DatasetQueryContext


@pytest.fixture()
def snapshot_dir(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "hello.py").write_text("print('hi')\nvalue = 1\n", encoding="utf-8")
    return repo


@pytest.fixture()
def tool(snapshot_dir: Path, tmp_path: Path) -> DatasetLogTool:
    ctx = DatasetQueryContext(
        query_id="q1",
        query="find hi",
        repo_url="https://example.com/repo.git",
        branch="main",
        commit="deadbeef",
        snapshot_path=snapshot_dir,
    )
    return DatasetLogTool(context=ctx, artifacts_root=tmp_path, run_name="test")


def test_write_success(tool: DatasetLogTool, tmp_path: Path) -> None:
    result = tool.execute(path="src/hello.py", start_line=1, end_line=2, confidence=0.9)
    assert result["status"] == "success"
    data = result["data"]
    assert data["chunk"]["path"] == "src/hello.py"
    assert data["chunk"]["start_line"] == 1
    assert data["chunk"]["confidence"] == 0.9
    raw_root = tmp_path / "test"
    assert not raw_root.exists()


def test_duplicate_rejected(tool: DatasetLogTool) -> None:
    first = tool.execute(path="src/hello.py", start_line=1, end_line=1, confidence=0.8)
    assert first["status"] == "success"
    second = tool.execute(path="src/hello.py", start_line=1, end_line=1, confidence=0.8)
    assert second["status"] == "error"
    assert "already" in second["content"]


def test_invalid_range(tool: DatasetLogTool) -> None:
    resp = tool.execute(path="src/hello.py", start_line=2, end_line=1, confidence=0.5)
    assert resp["status"] == "error"
    assert "valid range" in resp["content"]


def test_out_of_bounds(tool: DatasetLogTool) -> None:
    resp = tool.execute(path="src/hello.py", start_line=1, end_line=50, confidence=0.5)
    assert resp["status"] == "error"
    assert "exceeds" in resp["content"]


def test_confidence_bounds(tool: DatasetLogTool) -> None:
    resp = tool.execute(path="src/hello.py", start_line=1, end_line=1, confidence=1.5)
    assert resp["status"] == "error"
    assert "confidence" in resp["content"]
