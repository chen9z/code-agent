from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from .models import GoldenChunk

SCHEMA_VERSION = "2025.11"


@dataclass
class DatasetSample:
    query_id: str
    query: str
    repo_url: str
    commit: str
    golden_chunks: List[GoldenChunk]
    schema_version: str = SCHEMA_VERSION


class DatasetBuilder:
    """Construct final dataset jsonl file."""

    def __init__(self, *, output_dir: str | Path, run_name: str | None = None) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or datetime.now(timezone.utc).strftime("%Y%m%d")
        self.dataset_path = self.output_dir / f"dataset_{self.run_name}.jsonl"

    def append(self, sample: DatasetSample) -> None:
        payload = {
            "schema_version": sample.schema_version,
            "query_id": sample.query_id,
            "query": sample.query,
            "repo": {
                "url": sample.repo_url,
                "commit": sample.commit,
            },
            "golden_chunks": [asdict(chunk) for chunk in sample.golden_chunks],
        }
        with self.dataset_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    def build_from(self, samples: Iterable[DatasetSample]) -> None:
        for sample in samples:
            self.append(sample)
