"""Render admin mail templates with ``{{key}}`` and ``{key}`` substitution."""

from __future__ import annotations

import html as html_lib
from typing import Any, Mapping


def render_template_string(
    template: str,
    values: Mapping[str, Any],
    *,
    escape_for_html: bool,
) -> str:
    """
    Replace ``{{name}}`` first, then ``{name}`` (longer keys first to reduce partial matches).
    Values are stringified; when ``escape_for_html`` is True, values are HTML-escaped.
    """
    out = str(template or "")
    items = sorted(
        ((str(k).strip(), v) for k, v in values.items() if str(k).strip()),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    for k, v in items:
        raw = "" if v is None else str(v)
        repl = html_lib.escape(raw, quote=False) if escape_for_html else raw
        out = out.replace("{{" + k + "}}", repl)
    for k, v in items:
        raw = "" if v is None else str(v)
        repl = html_lib.escape(raw, quote=False) if escape_for_html else raw
        out = out.replace("{" + k + "}", repl)
    return out
