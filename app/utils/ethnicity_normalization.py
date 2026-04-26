"""
Shared helpers for ethnicity string comparison and display.

``MongoDB`` may store the same concept under different surface forms, e.g.
``Middle Eastern`` vs ``middle-eastern`` vs ``middle_eastern``. We map every
non-empty string to a *semantic key* (lowercase, hyphens/underscores → spaces,
whitespace collapsed) for deduplication and filtering, and a *canonical label*
(smart title case, connectives like *or* / *and* lowercased in the middle) for API responses and stable storage.
"""

from __future__ import annotations

import re
from typing import Optional

_SEP = re.compile(r"[-_\s]+")

# In multi-word labels, do not title-case these when between other words
# (unlike string.capwords, so "X or Y" is not "X Or Y" — a common source of
# "missing" rows when grepping the DB for the API string).
_MINOR_CONNECTIVES = frozenset(
    {
        "a", "an", "and", "as", "at", "but", "by", "for", "if", "in", "is", "it",
        "its", "nor", "of", "off", "on", "or", "per", "so", "the", "to", "up",
        "vs", "via", "v",
    }
)


def _title_ethnicity_words(semantic_key: str) -> str:
    """AP-style label: first and last word capitalized; connectives in lower case."""
    words = [w for w in semantic_key.split(" ") if w]
    n = len(words)
    if n == 0:
        return semantic_key
    out: list[str] = []
    for i, w in enumerate(words):
        low = w.lower()
        is_edge = i == 0 or i == n - 1
        if is_edge or low not in _MINOR_CONNECTIVES:
            if len(low) <= 1:
                out.append(low.upper())
            else:
                out.append(low[0].upper() + low[1:])
        else:
            out.append(low)
    return " ".join(out)


def ethnicity_semantic_key(value: object) -> Optional[str]:
    """
    Return a stable comparable key, or None when value is empty/missing.
    "Middle Eastern", "middle-eastern" → "middle eastern"
    """
    if value is None:
        return None
    t = str(value).strip()
    if not t:
        return None
    parts = [p for p in _SEP.split(t.lower()) if p]
    if not parts:
        return None
    return " ".join(parts)


def ethnicity_canonical_label(value: object) -> Optional[str]:
    """
    One display form per semantic key, using word-boundary title case that keeps
    small connectives (or, and, of, …) lower case when between other words.
    """
    k = ethnicity_semantic_key(value)
    if not k:
        return None
    return _title_ethnicity_words(k)
