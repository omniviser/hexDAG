import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

import yaml

ErrorCode = Literal[
    "too_large",
    "too_deep",
    "invalid_syntax",
    "unrecoverable",
    "no_yaml_found",
]

logger = logging.getLogger(__name__)


@dataclass
class SafeYAMLResult:
    data: Any | None = None
    error: ErrorCode | None = None
    message: str | None = None
    line: int | None = None
    col: int | None = None
    preview: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class SafeYAML:
    def __init__(self, max_size_bytes: int = 1_000_000, max_depth: int = 20):
        self.max_size_bytes = max_size_bytes
        self.max_depth = max_depth

    def loads(self, data: str | bytes | bytearray) -> SafeYAMLResult:
        text = (
            data.decode("utf-8", errors="strict")
            if isinstance(data, (bytes, bytearray))
            else str(data)
        )

        if len(text.encode("utf-8")) > self.max_size_bytes:
            return SafeYAMLResult(error="too_large", message="YAML exceeds size limit")

        if self._estimate_depth(text) > self.max_depth:
            return SafeYAMLResult(error="too_deep", message="YAML exceeds depth limit")

        try:
            parsed = yaml.safe_load(text)
            return SafeYAMLResult(data=parsed)
        except yaml.YAMLError as e:
            return self._handle_yaml_error(e, text)
        except Exception as e:
            logger.exception("Unrecoverable YAML parse failure")
            return SafeYAMLResult(error="unrecoverable", message=str(e))

    def loads_from_text(self, text: str) -> SafeYAMLResult:
        candidate = self._extract_yaml(text)
        if not candidate:
            return SafeYAMLResult(error="no_yaml_found", message="No YAML found in text")
        return self.loads(candidate)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_yaml(text: str) -> str | None:
        """Extract YAML content from LLM responses or markdown fenced blocks."""
        if not text:
            return None

        # Prefer fenced code blocks first
        match = re.search(r"```ya?ml\s*([\s\S]*?)```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Generic fallback: content that looks like YAML (starts with key:)
        match = re.search(r"(^|\n)([A-Za-z0-9_-]+:)", text)
        if match:
            # Try to take the block until triple backticks or EOF
            segment = text[match.start() :]
            cutoff = re.split(r"```", segment)[0]
            return cutoff.strip()

        return None

    @staticmethod
    def _estimate_depth(text: str) -> int:
        """Approximate YAML nesting by counting indentation changes."""
        depth = 0
        max_depth = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            level = indent // 2  # roughly 2 spaces per level
            depth = max(depth, level)
            max_depth = max(max_depth, depth)
        return max_depth

    @staticmethod
    def _format_error_line(text: str, line_no: int, col_no: int, context: int = 1) -> str | None:
        """Return snippet of the line with a caret pointing to error column."""
        lines = text.splitlines()
        if 1 <= line_no <= len(lines):
            line = lines[line_no - 1]
            caret_line = " " * (col_no - 1) + "^"
            return f"{line}\n{caret_line}"
        return None

    def _handle_yaml_error(self, err: yaml.YAMLError, text: str) -> SafeYAMLResult:
        if isinstance(err, yaml.MarkedYAMLError) and err.problem_mark is not None:
            mark = err.problem_mark
            preview = self._format_error_line(text, mark.line + 1, mark.column + 1)
            return SafeYAMLResult(
                error="invalid_syntax",
                message=str(err),
                line=mark.line + 1,
                col=mark.column + 1,
                preview=preview,
            )
        return SafeYAMLResult(error="invalid_syntax", message=str(err))
