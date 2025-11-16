"""Prompt assets for different agent flavors."""

from pathlib import Path

__all__ = ["DATASET_SYSTEM_PROMPT"]

_DATASET_PROMPT_PATH = Path(__file__).with_name("dataset.md")
DATASET_SYSTEM_PROMPT = _DATASET_PROMPT_PATH.read_text(encoding="utf-8").strip()
