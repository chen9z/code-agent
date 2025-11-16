from __future__ import annotations

import json

from adapters.workspace.tree_sitter.parser import TagKind, TreeSitterProjectParser


def test_tree_sitter_parser_extracts_python_symbols(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    source = project_root / "example.py"
    source.write_text(
        """
import math


def area(radius: float) -> float:
    return math.pi * radius ** 2

value = area(3.0)
""".strip()
    )

    with TreeSitterProjectParser() as parser:
        symbols = parser.parse_project(project_root)
        assert symbols, "expected at least one symbol extracted"
        names_and_kinds = {(s.name, s.kind) for s in symbols}
        assert ("area", TagKind.DEF) in names_and_kinds

        symbol = next(sym for sym in symbols if sym.name == "area")
        assert "code_snippet" in symbol.metadata
        assert symbol.metadata["code_snippet"].strip().startswith("def area")
        assert "references" in symbol.metadata
        payload = symbol.to_payload("demo_project", str(project_root))
        assert payload["project_name"] == "demo_project"
        assert payload["metadata"]["code_snippet"].strip().startswith("def area")
        point_id = symbol.point_id("demo_project")
        assert len(point_id) == 36

        export_path = tmp_path / "symbols.jsonl"
        parser.export_symbols(symbols, export_path)

    assert export_path.exists()
    exported = [json.loads(line) for line in export_path.read_text().splitlines()]
    assert any(entry["name"] == "area" for entry in exported)
    assert all("relative_path" in entry for entry in exported)
    assert any("metadata" in entry and "code_snippet" in entry["metadata"] for entry in exported)


def test_tree_sitter_parser_logs_on_parse_failure(tmp_path, monkeypatch, caplog):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    source = project_root / "broken.py"
    source.write_text("def broken(:\n    pass\n")

    class BoomParser:
        def parse(self, _bytes):
            raise RuntimeError("boom")

    with TreeSitterProjectParser() as parser:
        monkeypatch.setattr(parser, "_get_parser", lambda _lang: BoomParser())
        caplog.set_level("WARNING")
        symbols = parser.parse_file(source, project_root=project_root)

    assert symbols == []
    assert any("Failed to parse" in record.message for record in caplog.records)
