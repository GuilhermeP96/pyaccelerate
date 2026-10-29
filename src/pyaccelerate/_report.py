"""Content-sized terminal reports shared by the engine and tuning commands."""

from __future__ import annotations

import re
import shutil
import unicodedata
from collections.abc import Iterable

from wcwidth import wcswidth

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def visual_width(text: str) -> int:
    """Count terminal cells, excluding ANSI escapes and combining characters."""
    return max(0, wcswidth(_ANSI.sub("", text)))


def _wrap(text: str, width: int) -> list[str]:
    # Keep escape sequences together. Measuring the whole candidate also handles
    # combining accents, wide characters and joined emoji sequences.
    raw = re.findall(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|.", text)
    tokens: list[str] = []
    for token in raw:
        if (
            tokens
            and not token.startswith("\x1b")
            and (
                unicodedata.combining(token)
                or token in ("\ufe0e", "\ufe0f", "\u200d")
                or tokens[-1].endswith("\u200d")
                or "\U0001f3fb" <= token <= "\U0001f3ff"
            )
        ):
            tokens[-1] += token
        else:
            tokens.append(token)
    result: list[str] = []
    style = ""
    while tokens:
        end = 0
        last_space = 0
        current = ""
        for index, token in enumerate(tokens):
            candidate = current + token
            if visual_width(candidate) > width:
                break
            current = candidate
            end = index + 1
            if token == " " and _ANSI.sub("", current).strip():
                last_space = end
        if end == len(tokens):
            result.append(style + current.rstrip())
            break
        cut = last_space or end or 1
        piece = "".join(tokens[:cut]).rstrip()
        result.append(style + piece)
        for escape in re.findall(r"\x1b\[[0-9;]*m", piece):
            if escape in ("\x1b[m", "\x1b[0m"):
                style = ""
            else:
                style += escape
        tokens = tokens[cut:]
        while tokens and tokens[0] == " ":
            tokens.pop(0)
    return result or [""]


def render_report(title: str, rows: Iterable[str | None], *, width: int | None = None) -> str:
    """Render aligned labels, wrapped values and a closed border on every row.

    ``None`` separates sections. Width includes borders and padding; by default
    reports fit the terminal (80 columns when output is redirected).
    """
    rows = list(rows)
    label_widths: dict[bool, int] = {}
    for row in rows:
        if row is not None:
            for line in row.expandtabs(4).splitlines():
                if ":" in line:
                    nested = line.startswith(" ")
                    label = line.lstrip().split(":", 1)[0]
                    label_widths[nested] = max(label_widths.get(nested, 0), visual_width(label))
    aligned: list[str | None] = []
    for row in rows:
        if row is None:
            aligned.append(None)
            continue
        for line in row.expandtabs(4).splitlines() or [""]:
            if ":" in line:
                nested = line.startswith(" ")
                label, value = line.lstrip().split(":", 1)
                padding = label_widths[nested] - visual_width(label) + 1
                line = ("  " if nested else "") + label + ":" + " " * padding + value.lstrip()
            aligned.append(line)
    limit = max(6, width if width is not None else shutil.get_terminal_size((80, 24)).columns)
    content_width = min(
        max([2, visual_width(title), *(visual_width(r) for r in aligned if r is not None)]),
        limit - 4,
    )

    def border(left: str, right: str) -> str:
        return left + "═" * (content_width + 2) + right

    def cell(text: str, *, center: bool = False) -> str:
        padding = content_width - visual_width(text)
        left = padding // 2 if center else 0
        # A color in a value must never color the padding/border.
        reset = "\x1b[0m" if "\x1b" in text else ""
        return "║ " + " " * left + text + reset + " " * (padding - left) + " ║"

    lines = [border("╔", "╗")]
    lines.extend(cell(line, center=True) for line in _wrap(title, content_width))
    lines.append(border("╠", "╣"))
    for row in aligned:
        if row is None:
            lines.append(border("╠", "╣"))
        else:
            lines.extend(cell(line) for line in _wrap(row, content_width))
    lines.append(border("╚", "╝"))
    return "\n".join(lines)
