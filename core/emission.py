from __future__ import annotations

"""Shared utilities for streaming structured agent output."""

from collections.abc import Mapping, Sequence
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


class EmitEvent(str):
    """String-compatible event object that carries structured payload data."""

    __slots__ = ("kind", "body", "payload")

    kind: str
    body: str
    payload: Any

    def __new__(
        cls,
        kind: str,
        body: str = "",
        *,
        payload: Any = None,
    ) -> "EmitEvent":
        normalized_kind = (kind or "").strip()
        normalized_body = body or ""
        display_entries = _extract_display(payload)
        composed_body = _compose_body(normalized_body, display_entries)

        if normalized_kind:
            prefix = f"[{normalized_kind}]"
            text = f"{prefix} {composed_body}" if composed_body else prefix
        else:
            text = composed_body

        event = str.__new__(cls, text)
        event.kind = normalized_kind
        event.body = normalized_body
        event.payload = payload
        return event

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the event."""

        return {
            "kind": self.kind,
            "body": self.body,
            "payload": self.payload,
            "display": list(self.display),
            "text": str(self),
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"EmitEvent(kind={self.kind!r}, body={self.body!r}, payload={self.payload!r})"
        )

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

    if isinstance(raw_display, Mapping):
        candidates = raw_display.items()
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
