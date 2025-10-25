from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ToolOutputEntry:
    call_id: str
    label: str
    status: str
    arguments: Dict[str, Any]
    output: str
    truncated: bool


class ToolOutputStore:
    """In-memory buffer that retains recent tool outputs for CLI inspection."""

    def __init__(self, max_entries: int = 50) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._entries: Dict[str, ToolOutputEntry] = {}
        self._order: List[str] = []
        self._last_truncated: Optional[str] = None

    def record(
        self,
        *,
        call_id: str,
        label: str,
        status: str,
        arguments: Dict[str, Any],
        output: str,
        truncated: bool,
    ) -> None:
        entry = ToolOutputEntry(
            call_id=call_id,
            label=label,
            status=status,
            arguments=dict(arguments),
            output=output,
            truncated=truncated,
        )
        if call_id in self._entries:
            self._order = [existing for existing in self._order if existing != call_id]
        self._entries[call_id] = entry
        self._order.append(call_id)
        if len(self._order) > self.max_entries:
            oldest = self._order.pop(0)
            self._entries.pop(oldest, None)
        if truncated:
            self._last_truncated = call_id

    def get(self, call_id: str) -> Optional[ToolOutputEntry]:
        return self._entries.get(call_id)

    def latest(self, *, truncated_only: bool = False) -> Optional[ToolOutputEntry]:
        if truncated_only and self._last_truncated:
            entry = self._entries.get(self._last_truncated)
            if entry is not None:
                return entry
        for call_id in reversed(self._order):
            entry = self._entries.get(call_id)
            if entry is None:
                continue
            if truncated_only and not entry.truncated:
                continue
            return entry
        return None

    def all(self) -> List[ToolOutputEntry]:
        return [self._entries[call_id] for call_id in self._order if call_id in self._entries]

    def clear(self) -> None:
        self._entries.clear()
        self._order.clear()
        self._last_truncated = None

