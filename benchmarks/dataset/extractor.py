from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import List

from .models import GoldenChunk


@dataclass
class ExtractResult:
    query_id: str
    chunks: List[GoldenChunk]
    errors: List[str]


class RawSampleExtractor:
    """Aggregate per-query raw_samples into deduped golden chunks."""

    def __init__(self, *, raw_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir).expanduser().resolve()

    def extract(self, query_id: str) -> ExtractResult:
        path = self.raw_dir / f"{query_id}.jsonl"
        if not path.exists():
            return ExtractResult(query_id=query_id, chunks=[], errors=["raw sample missing"])

        seen: set[str] = set()
        chunks: List[GoldenChunk] = []
        errors: List[str] = []

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    errors.append("json decode error")
                    continue
                chunk = payload.get("chunk") or {}
                fingerprint = chunk.get("fingerprint")
                if not fingerprint or fingerprint in seen:
                    continue
                seen.add(fingerprint)
                content = chunk.get("content", "")
                content_hash = sha1(content.encode("utf-8")).hexdigest()
                chunks.append(
                    GoldenChunk(
                        path=chunk.get("path", ""),
                        start_line=int(chunk.get("start_line", 0)),
                        end_line=int(chunk.get("end_line", 0)),
                        confidence=float(chunk.get("confidence", 0)),
                        content=content,
                        fingerprint=fingerprint,
                        content_hash=content_hash,
                    )
                )
        return ExtractResult(query_id=query_id, chunks=chunks, errors=errors)
