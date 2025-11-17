from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseTool(ABC):
    """Abstract base class for all tools in the code-agent system."""

    DEFAULT_MAX_CHARS = 50_000
    DEFAULT_MAX_LINE_LENGTH = 2_000
    _TRUNCATION_SUFFIX = "… (truncated)"

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters."""
        pass

    def clip_text(
        self,
        text: Any,
        *,
        max_chars: Optional[int] = None,
        max_line_length: Optional[int] = None,
    ) -> Tuple[str, bool]:
        """Clamp text output to sane defaults.

        Returns a tuple of (clipped_text, was_truncated).
        - max_chars limits the total output length; defaults to DEFAULT_MAX_CHARS.
        - max_line_length limits any individual line (before newline characters);
          defaults to DEFAULT_MAX_LINE_LENGTH.
        """

        if text is None:
            return "", False

        raw_text = text if isinstance(text, str) else str(text)
        if not raw_text:
            return "", False

        total_limit = self.DEFAULT_MAX_CHARS if not max_chars or max_chars <= 0 else int(max_chars)
        line_limit = (
            self.DEFAULT_MAX_LINE_LENGTH if not max_line_length or max_line_length <= 0 else int(max_line_length)
        )

        clipped_parts: list[str] = []
        consumed = 0
        truncated = False

        for raw_line in raw_text.splitlines(keepends=True):
            if consumed >= total_limit:
                truncated = True
                break

            # Preserve line endings while clipping the textual portion.
            line_content = raw_line.rstrip("\r\n")
            line_ending = raw_line[len(line_content) :]

            display_line = line_content
            if len(line_content) > line_limit:
                display_line = line_content[:line_limit] + self._TRUNCATION_SUFFIX
                truncated = True

            segment = display_line + line_ending
            segment_len = len(segment)

            if consumed + segment_len <= total_limit:
                clipped_parts.append(segment)
                consumed += segment_len
                continue

            available = max(total_limit - consumed, 0)
            if available > 0:
                clipped_parts.append(segment[:available])
            truncated = True
            break

        result = "".join(clipped_parts)
        if truncated and not result.endswith(self._TRUNCATION_SUFFIX):
            result = result.rstrip("\r\n") + self._TRUNCATION_SUFFIX

        return result, truncated

    def __call__(self, **kwargs) -> Any:
        """Make the tool callable."""
        return self.execute(**kwargs)
