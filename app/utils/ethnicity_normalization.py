"""
Shared helpers for ethnicity string comparison and display.

``MongoDB`` may store the same concept under different surface forms, e.g.
``Middle Eastern`` vs ``middle-eastern`` vs ``middle_eastern``. We map every
non-empty string to a *semantic key* (lowercase, hyphens/underscores → spaces,
whitespace collapsed) for deduplication and filtering, and a *canonical label*
(``str.title``-style word boundaries) for API responses and stable storage.
"""

from __future__ import annotations

import re
import string
from typing import Optional

_SEP = re.compile(r"[-_\s]+")


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
    One display form per semantic key, using capwords on words split by
    our separator pattern (already normalized via semantic key).
    """
    k = ethnicity_semantic_key(value)
    if not k:
        return None
    return string.capwords(k)
