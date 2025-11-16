from __future__ import annotations

"""Tree-sitter based project parsing utilities.

This module adapts portions of the Locify project's tree-sitter parser
implementation (https://github.com/ryanhoangt/locify) under the MIT License.
"""

import json
import logging
import hashlib
import uuid
from collections import defaultdict
from dataclasses import dataclass, asdict, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from grep_ast import filename_to_lang
from tree_sitter import Parser, Query, QueryCursor

from retrieval.splitter import iter_repository_files
from adapters.workspace.tree_sitter.language_loader import (
    is_language_supported,
    load_language,
)


logger = logging.getLogger(__name__)


class TagKind(str, Enum):
    """Kinds of symbol tags discovered via tree-sitter queries."""

    DEF = "def"
    REF = "ref"


@dataclass(frozen=True)
class ParsedSymbol:
    """Symbol metadata extracted from source code."""

    relative_path: str
    absolute_path: str
    language: str
    start_line: int
    end_line: int
    name: str
    kind: TagKind
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ParsedSymbol":
        return cls(
            relative_path=data["relative_path"],
            absolute_path=data["absolute_path"],
            language=data["language"],
            start_line=int(data["start_line"]),
            end_line=int(data["end_line"]),
            name=data["name"],
            kind=TagKind(data["kind"]),
            metadata=data.get("metadata", {}),
        )

    def point_id(self, project_name: str) -> str:
        """Stable UUID derived from project + location."""

        base = f"{project_name}:{self.relative_path}:{self.start_line}-{self.end_line}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

    def to_payload(self, project_name: str, project_path: str) -> Dict[str, Any]:
        """Build payload dictionary for vector store insertion."""

        payload: Dict[str, Any] = {
            "project_name": project_name,
            "project_path": project_path,
            "language": self.language,
            "relative_path": self.relative_path,
            "absolute_path": self.absolute_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol_name": self.name,
            "symbol_kind": self.kind.value,
            "symbol_id": f"{project_name}:{self.relative_path}:{self.start_line}-{self.end_line}",
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class TreeSitterProjectParser:
    """Parse repositories with tree-sitter queries to extract symbol tags."""

    def __init__(
        self,
        *,
        queries_dir: Optional[Path | str] = None,
    ) -> None:
        base_dir = Path(queries_dir) if queries_dir else Path(__file__).resolve().parent / "queries"
        self.queries_dir = base_dir
        self._language_cache: dict[str, object] = {}
        self._parser_cache: dict[str, object] = {}

    def close(self) -> None:
        # 保留接口以兼容上下文管理器，当前无持久化缓存资源需要关闭。
        pass

    # Context manager support if needed
    def __enter__(self) -> "TreeSitterProjectParser":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def parse_project(self, project_root: Path | str) -> List[ParsedSymbol]:
        """Parse all supported files under project_root."""
        root = Path(project_root).expanduser().resolve()
        symbols: List[ParsedSymbol] = []
        for file_path in iter_repository_files(root):
            symbols.extend(self.parse_file(file_path, project_root=root))
        return symbols

    def parse_file(self, file_path: Path | str, *, project_root: Optional[Path] = None) -> List[ParsedSymbol]:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return []
        root = project_root or path.parent
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        return self._parse_file_uncached(path, relative)

    def export_symbols(self, symbols: Sequence[ParsedSymbol], output_path: Path | str) -> Path:
        """Write parsed symbols to JSONL for downstream consumption."""
        out_path = Path(output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fout:
            for symbol in symbols:
                fout.write(json.dumps(symbol.to_dict(), ensure_ascii=False))
                fout.write("\n")
        return out_path

    # Internal helpers -----------------------------------------------------------------
    def _parse_file_uncached(self, path: Path, relative: str) -> List[ParsedSymbol]:
        lang = filename_to_lang(str(path))
        if not lang or not is_language_supported(lang):
            return []

        query_path = self.queries_dir / f"tree-sitter-{lang}-tags.scm"
        if not query_path.exists():
            return []

        try:
            code = path.read_text(encoding="utf-8")
        except Exception:
            code = path.read_text(encoding="utf-8", errors="ignore")
        if not code.strip():
            return []

        try:
            parser = self._get_parser(lang)
            language = self._get_language(lang)
        except (ImportError, AttributeError, KeyError):
            return []
        code_bytes = code.encode("utf-8")
        try:
            parsed_tree = parser.parse(code_bytes)
        except Exception as exc:
            logger.warning(
                "Failed to parse %s with tree-sitter, possibly due to a syntax error. Details: %s",
                path,
                exc,
            )
            return []

        query = Query(language, query_path.read_text())
        cursor = QueryCursor(query)
        captures = cursor.captures(parsed_tree.root_node)

        def decode_slice(start_byte: int, end_byte: int) -> str:
            return code_bytes[start_byte:end_byte].decode("utf-8", errors="ignore")

        def build_scope(node) -> List[str]:
            scope: List[str] = []
            current = node.parent
            while current is not None:
                name_node = current.child_by_field_name("name")
                if name_node is not None:
                    scope_name = decode_slice(name_node.start_byte, name_node.end_byte).strip()
                    if scope_name:
                        scope.append(scope_name)
                current = current.parent
            scope.reverse()
            return scope

        references: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        definitions: List[ParsedSymbol] = []

        for capture_name, nodes in captures.items():
            kind = self._map_tag_to_kind(capture_name)
            if kind is None:
                continue
            for node in nodes:
                name = decode_slice(node.start_byte, node.end_byte).strip()
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                if kind is TagKind.DEF:
                    body_node = node.parent or node
                    body_start_line = body_node.start_point[0] + 1
                    body_end_line = body_node.end_point[0] + 1
                    code_snippet = decode_slice(body_node.start_byte, body_node.end_byte)
                    identifier_text = decode_slice(node.start_byte, node.end_byte)
                    metadata: Dict[str, Any] = {
                        "code_snippet": code_snippet,
                        "identifier": identifier_text,
                        "scope_path": build_scope(node),
                        "identifier_range": {
                            "start_line": start_line,
                            "end_line": end_line,
                            "start_column": node.start_point[1] + 1,
                            "end_column": node.end_point[1] + 1,
                        },
                        "byte_range": [body_node.start_byte, body_node.end_byte],
                        "source_hash": hashlib.sha256(
                            code_bytes[body_node.start_byte : body_node.end_byte]
                        ).hexdigest(),
                        "references": [],
                    }
                    docstring = self._extract_docstring(code_snippet, lang)
                    if docstring:
                        metadata["docstring"] = docstring
                    definitions.append(
                        ParsedSymbol(
                            relative_path=relative,
                            absolute_path=str(path),
                            language=lang,
                            start_line=body_start_line,
                            end_line=body_end_line,
                            name=name,
                            kind=kind,
                            metadata=metadata,
                        )
                    )
                else:  # TagKind.REF
                    references[name].append(
                        {
                            "relative_path": relative,
                            "absolute_path": str(path),
                            "start_line": start_line,
                            "end_line": end_line,
                            "text": decode_slice(node.start_byte, node.end_byte),
                            "scope_path": build_scope(node),
                        }
                    )

        enriched: List[ParsedSymbol] = []
        for symbol in definitions:
            meta = dict(symbol.metadata)
            name_refs = references.get(symbol.name)
            if name_refs:
                meta["references"] = name_refs
            enriched.append(replace(symbol, metadata=meta))

        enriched.sort(key=lambda sym: (sym.start_line, sym.end_line, sym.name))
        return enriched

    def _map_tag_to_kind(self, tag_value: str) -> Optional[TagKind]:
        if tag_value.startswith("name.definition."):
            return TagKind.DEF
        if tag_value.startswith("name.reference."):
            return TagKind.REF
        return None

    def _extract_docstring(self, code_snippet: str, language: str) -> Optional[str]:
        """Best-effort docstring extraction for supported languages."""

        if language != "python":
            return None
        lines = code_snippet.splitlines()
        if len(lines) < 2:
            return None
        first_body_idx = 1
        while first_body_idx < len(lines) and not lines[first_body_idx].strip():
            first_body_idx += 1
        if first_body_idx >= len(lines):
            return None
        candidate = lines[first_body_idx].lstrip()
        if not candidate.startswith(('"""', "'''")):
            return None
        delimiter = candidate[:3]
        content = candidate[3:]
        doc_lines: List[str] = []
        if content.endswith(delimiter) and len(content) > 3:
            doc_lines.append(content[:-3])
            return "\n".join(doc_lines).strip() or None
        if content:
            doc_lines.append(content)
        for line in lines[first_body_idx + 1 :]:
            stripped = line.strip()
            if stripped.endswith(delimiter):
                doc_lines.append(stripped[:-3])
                break
            doc_lines.append(stripped)
        doc = "\n".join(doc_lines).strip()
        return doc or None

    def _get_parser(self, language: str) -> Parser:
        parser = self._parser_cache.get(language)
        if parser is None:
            parser = Parser()
            parser.language = self._get_language(language)
            self._parser_cache[language] = parser
        return parser

    def _get_language(self, language: str):
        lang = self._language_cache.get(language)
        if lang is None:
            lang = load_language(language)
            self._language_cache[language] = lang
        return lang


def parse_project_symbols(project_root: Path | str) -> List[ParsedSymbol]:
    """Convenience helper to parse symbols for a project in one call."""
    parser = TreeSitterProjectParser()
    try:
        return parser.parse_project(project_root)
    finally:
        parser.close()
