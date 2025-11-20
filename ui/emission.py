from __future__ import annotations

"""Shared utilities for streaming structured agent output."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple, Union

DisplayItem = Tuple[str, Union[str, None]]
OutputMessage = Union[str, "EmitEvent"]
OutputCallback = Callable[[OutputMessage], None]

__all__ = [
    "EmitEvent",
    "DisplayItem",
    "OutputCallback",
    "OutputMessage",
    "create_emit_event",
]


@dataclass
class EmitEvent:
    """Structured event object that carries payload data."""

    kind: str
    body: str = ""
    payload: Any = field(default=None)

    def __str__(self) -> str:
        normalized_kind = (self.kind or "").strip()
        normalized_body = self.body or ""
        display_entries = _extract_display(self.payload)
        composed_body = _compose_body(normalized_body, display_entries)

        if normalized_kind:
            prefix = f"[{normalized_kind}]"
            return f"{prefix} {composed_body}" if composed_body else prefix
        return composed_body

    @property
    def display(self) -> Tuple[DisplayItem, ...]:
        """Return display-friendly key/value tuples derived from the payload."""
        return _extract_display(self.payload)


def create_emit_event(
    kind: str,
    body: str = "",
    *,
    payload: Any = None,
) -> EmitEvent:
    """Helper to mirror the `[tag] message` convention with a single payload field."""
    return EmitEvent(kind, body, payload=payload)


def _extract_display(payload: Any) -> Tuple[DisplayItem, ...]:
    if not isinstance(payload, Mapping):
        return ()

    raw_display = payload.get("display")
    if raw_display is None:
        return ()

    candidates: Iterable[Any]
    if isinstance(raw_display, Mapping):
        candidates = raw_display.items()
    elif isinstance(raw_display, str):
        candidates = (("result", raw_display),)
    elif isinstance(raw_display, Sequence) and not isinstance(raw_display, (str, bytes)):
        candidates = raw_display
    else:
        candidates = (raw_display,)

    normalized: list[DisplayItem] = []
    for entry in candidates:
        if isinstance(entry, Mapping):
            for key, value in entry.items():
                _append_display(normalized, key, value)
            continue

        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            if not entry:
                continue
            key = entry[0]
            value = entry[1] if len(entry) > 1 else None
            _append_display(normalized, key, value)
            continue

        _append_display(normalized, entry, None)

    return tuple(normalized)


def _append_display(container: list[DisplayItem], key: Any, value: Any) -> None:
    name = str(key or "").strip()
    if not name:
        return

    if value is None:
        container.append((name, None))
        return

    text = str(value).strip()
    if not text:
        container.append((name, None))
        return
    container.append((name, text))


def _compose_body(body: str, metadata: Tuple[DisplayItem, ...]) -> str:
    segments: list[str] = []
    if body:
        segments.append(body)
    for key, value in metadata:
        if value is None:
            segments.append(key)
        else:
            segments.append(f"{key}: {value}")
    return " | ".join(segment for segment in segments if segment).strip()
