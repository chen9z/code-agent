from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

from rich.console import Console
from rich.text import Text

_RICH_STYLE_MAP = {
    "assistant": "white",
    "assistant:planner": "white",
    "planner": "white",
    "plan": "green",
    "tool": "green",
    "tool-output": "dim",
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
) -> Callable[[Any], None]:
    """Return a callback that renders agent events using a Rich console."""

    active_console = console or Console()

    def emit(message: Any) -> None:
        if message is None:
            return
        if isinstance(message, Mapping):
            text = stringify(message)
        else:
            text = message if isinstance(message, str) else stringify(message)
        tag, body = _split_message_tag(text)
        if tag is None:
            active_console.print(Text(str(text)))
            active_console.print()
            return

        normalized_tag = tag.lower()
        if normalized_tag == "user":
            line = Text.assemble(
                Text("> ", style="bold white"),
                Text(body, style="bold white"),
            )
            active_console.print(line)
            active_console.print()
            return

        if normalized_tag == "system":
            style = _RICH_STYLE_MAP.get(normalized_tag, "magenta")
            active_console.print(Text(body, style=style))
            active_console.print()
            return

        if normalized_tag in {"assistant", "assistant:planner", "planner"}:
            _render_bullet(active_console, body, [], "white", "white")
            return

        if normalized_tag == "plan":
            return

        if normalized_tag == "tool":
            header, metadata = _parse_structured_body(body)
            status = _extract_metadata_value(metadata, "status") or "success"
            bullet_style = "green" if status.lower() == "success" else "red"
            header_style = "bold green" if status.lower() == "success" else "bold red"
            _render_bullet(active_console, header, metadata, bullet_style, header_style)
            return

        if normalized_tag == "tool-output":
            style = _RICH_STYLE_MAP.get(normalized_tag, "dim")
            active_console.print(Text(body, style=style))
            active_console.print()
            return

        style = _RICH_STYLE_MAP.get(normalized_tag, "white")
        _render_bullet(active_console, body, [], style, style)

    return emit


def _split_message_tag(message: str) -> tuple[Optional[str], str]:
    stripped = message.strip()
    if not stripped.startswith("["):
        return None, message
    closing = stripped.find("]")
    if closing <= 1:
        return None, message
    suffix = stripped[closing + 1 :]
    if not suffix.startswith(" "):
        return None, message
    tag = stripped[1:closing]
    body = stripped[closing + 2 :]
    return tag, body


def _parse_structured_body(body: str) -> tuple[str, List[tuple[str, str]]]:
    parts = [segment.strip() for segment in body.split("|") if segment.strip()]
    if not parts:
        return body, []
    header = parts[0]
    metadata: List[tuple[str, str]] = []
    for segment in parts[1:]:
        if ":" not in segment:
            continue
        key, value = segment.split(":", 1)
        metadata.append((key.strip(), value.strip()))
    return header, metadata


def _extract_metadata_value(metadata: List[tuple[str, str]], key: str) -> Optional[str]:
    for meta_key, value in metadata:
        if meta_key.lower() == key.lower():
            return value
    return None


def _render_bullet(
    console: Console,
    header: str,
    metadata: List[tuple[str, str]],
    bullet_style: str,
    header_style: str,
) -> None:
    bullet = Text("● ", style=bullet_style)
    header_text = Text(header, style=header_style)
    console.print(Text.assemble(bullet, header_text))

    meta_lines = _format_metadata(metadata)
    for idx, line in enumerate(meta_lines):
        prefix = Text("└ " if idx == 0 else "  ", style="dim")
        console.print(Text.assemble(prefix, line))

    console.print()


def _format_metadata(metadata: List[tuple[str, str]]) -> List[Text]:
    lines: List[Text] = []
    for key, value in metadata:
        normalized = key.lower()
        value_text = value
        if not value_text:
            continue
        if normalized == "status":
            value_str = str(value_text)
            if value_str.lower() != "success":
                lines.append(Text(f"status: {value_str}", style="bold red"))
            continue
        if normalized == "args":
            line = Text(f"args: {value_text}", style="dim")
        elif normalized in {"output", "result"}:
            line = Text(value_text, style="white")
        elif normalized == "error":
            line = Text(f"error: {value_text}", style="bold red")
        else:
            line = Text(f"{key}: {value_text}", style="dim")
        lines.append(line)
    return lines
