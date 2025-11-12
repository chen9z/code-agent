from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.text import Text

from ui.emission import EmitEvent, OutputCallback
_RICH_STYLE_MAP = {
    "assistant": "white",
    "plan": "green",
    "tool": "green",
    "user": "bold white",
    "system": "magenta",
    "warning": "yellow",
}


def stringify_payload(value: Any) -> str:
    """Render values as compact strings suitable for console output."""

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return repr(value)


def preview_payload(value: Any, limit: int) -> str:
    text = stringify_payload(value)
    if limit <= 3 or len(text) <= limit:
        return text[:limit]
    return f"{text[: limit - 3]}..."


def create_rich_output(
    console: Optional[Console] = None,
    *,
    stringify: Callable[[Any], str] = stringify_payload,
) -> OutputCallback:
    """Return a callback that renders agent events using a Rich console."""

    active_console = console or Console()

    def emit(message: Any) -> None:
        if message is None:
            return
        display_entries: List[Tuple[str, Optional[str]]] = []
        if isinstance(message, EmitEvent):
            tag = message.kind or None
            body = message.body
            text = str(message)
            display_entries = list(message.display)
            if not display_entries and tag and tag.lower() == "tool":
                body, display_entries = _parse_structured_body(body)
        else:
            tag = "unknown"
            try:
                body = json.dumps(message, ensure_ascii=False)
            except TypeError:
                body = stringify(message)
            text = body
        tag = tag or None
        if tag is None:
            active_console.print(Text(str(text)))
            active_console.print()
            return

        kind = tag.strip().lower()
        if not kind:
            active_console.print(Text(str(text)))
            active_console.print()
            return

        if kind == "user":
            line = Text.assemble(
                Text("> ", style="bold white"),
                Text(body, style="bold white"),
            )
            active_console.print(line)
            active_console.print()
            return

        if kind == "system":
            style = _RICH_STYLE_MAP.get(kind, "magenta")
            active_console.print(Text(body, style=style))
            active_console.print()
            return

        if kind == "assistant":
            _render_bullet(active_console, body, [], "white", "white")
            return

        if kind == "tool":
            payload = message.payload if isinstance(message, EmitEvent) else None
            header = _format_tool_header(body, payload)
            if not display_entries:
                header, display_entries = _parse_structured_body(body)
            status = (
                (payload or {}).get("status")
                or _extract_display_value(display_entries, "status")
                or "success"
            )
            bullet_style = "green" if status.lower() == "success" else "red"
            header_style = "bold green" if status.lower() == "success" else "bold red"
            _render_bullet(active_console, header, display_entries, bullet_style, header_style)
            return

        style = _RICH_STYLE_MAP.get(kind, "white")
        _render_bullet(active_console, body, [], style, style)

    return emit


def _parse_structured_body(body: str) -> tuple[str, List[tuple[str, Optional[str]]]]:
    parts = [segment.strip() for segment in body.split("|") if segment.strip()]
    if not parts:
        return body, []
    header = parts[0]
    metadata: List[tuple[str, Optional[str]]] = []
    for segment in parts[1:]:
        if ":" not in segment:
            metadata.append((segment.strip(), None))
            continue
        key, value = segment.split(":", 1)
        metadata.append((key.strip(), value.strip()))
    return header, metadata


def _extract_display_value(metadata: List[tuple[str, Optional[str]]], key: str) -> Optional[str]:
    for meta_key, value in metadata:
        if meta_key.lower() == key.lower():
            return value
    return None


def _render_bullet(
    console: Console,
    header: str,
    metadata: Sequence[tuple[str, Optional[str]]],
    bullet_style: str,
    header_style: str,
) -> None:
    bullet = Text("● ", style=bullet_style)
    header_text = Text(header, style=header_style)
    console.print(Text.assemble(bullet, header_text))

    meta_lines = _format_display(metadata)
    for idx, line in enumerate(meta_lines):
        prefix = Text("└ " if idx == 0 else "  ", style="dim")
        console.print(Text.assemble(prefix, line))

    console.print()


def _format_display(metadata: Sequence[tuple[str, Optional[str]]]) -> List[Text]:
    lines: List[Text] = []
    for key, value in metadata:
        normalized = key.lower()
        value_text = str(value) if value is not None else ""
        if not value_text and normalized not in {"note"}:
            continue
        if normalized == "status":
            value_str = str(value_text)
            if value_str.lower() != "success":
                lines.append(Text(f"status: {value_str}", style="bold red"))
            continue
        if normalized == "args":
            text_value = f"args: {value_text}"
            style = "dim"
        elif normalized in {"output", "result", "content", "match", "todo"}:
            text_value = value_text
            style = "white"
        elif normalized == "error":
            text_value = f"error: {value_text}"
            style = "bold red"
        elif normalized == "note":
            text_value = value_text or key
            style = "dim"
        else:
            text_value = f"{key}: {value_text}"
            style = "dim"

        for segment in _split_multiline_text(text_value):
            lines.append(Text(segment, style=style))
    return lines


def _format_tool_header(body: str, payload: Optional[dict[str, Any]]) -> str:
    if not isinstance(payload, dict):
        return body
    tool_key = str(payload.get("tool") or "").lower()
    if tool_key == "todo_write":
        return body
    arguments = payload.get("arguments")
    if not arguments:
        return body
    args_preview = preview_payload(arguments, 80)
    if args_preview in {"", "{}", "null"}:
        return body
    return f"{body}({args_preview})"


def _split_multiline_text(value: str) -> List[str]:
    if value == "":
        return [""]
    segments = value.splitlines()
    return segments or [value]
