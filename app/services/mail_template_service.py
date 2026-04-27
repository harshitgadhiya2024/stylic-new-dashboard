"""Render ``{{ var }}`` placeholders in admin mail templates; escape values for HTML."""

from __future__ import annotations

import html
import re
from typing import Any, Dict

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def render_template_string(
    template: str,
    data: Dict[str, Any],
    *,
    is_html: bool = True,
) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        raw = data.get(key, "")
        if raw is None:
            raw = ""
        s = str(raw)
        if is_html:
            return html.escape(s, quote=True)
        return s

    return _PLACEHOLDER.sub(repl, template or "")
