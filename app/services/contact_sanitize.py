"""
Strict plaintext rules for the public contact-sales endpoint.

Rejects obvious HTML/JS/injection patterns and strips dangerous characters.
Output is safe to store and to echo in admin UIs (still escape on render).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Block common script / template / code injection patterns (case-insensitive)
_SUSPICIOUS = re.compile(
    r"<\s*script|</\s*script|</?\s*iframe|</?\s*object|</?\s*embed|"
    r"javascript\s*:|data\s*:\s*text/html|on\w+\s*=|"
    r"eval\s*\(|expression\s*\(|import\s+[\w.]+|require\s*\(|__proto__|"
    r"react\.|jsx|dangerouslysetinnerhtml|document\.|window\.|"
    r"<\?php|<\?=",
    re.IGNORECASE | re.DOTALL,
)

# Phone: optional, digits, spaces, common punctuation
_PHONE_OK = re.compile(r"^[\d\s\-\+\(\)]+$")

# Normal field: letters, numbers, space, and a small set of punctuation (names, job titles)
_PLAIN_NAME_OK = re.compile(
    r"^[\w\s\-'’.,&()/@+°]+$",
    re.UNICODE,
)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def sanitize_plain_line(s: str, max_len: int) -> str:
    """One-line fields: no newlines, no angle brackets, Unicode letters ok."""
    t = _nfc(s)[:max_len]
    if not t:
        raise ValueError("Field is empty")
    if any(c in t for c in "\n\r\t"):
        raise ValueError("Invalid line breaks in field")
    if "<" in t or ">" in t:
        raise ValueError("Invalid characters: angle brackets are not allowed")
    if _SUSPICIOUS.search(t):
        raise ValueError("Content not allowed in this field")
    if not _PLAIN_NAME_OK.match(t):
        raise ValueError("Unsupported characters in this field")
    return t


def sanitize_optional_phone(s: Optional[str]) -> Optional[str]:
    if s is None or not str(s).strip():
        return None
    t = _nfc(s)[:32]
    if "<" in t or ">" in t or _SUSPICIOUS.search(t):
        raise ValueError("Invalid phone value")
    if not _PHONE_OK.match(t):
        raise ValueError("Invalid phone format")
    return t


def sanitize_message(s: str, max_len: int) -> str:
    """
    Multi-line message: no HTML tags, no code-like payloads; newlines allowed.
    """
    t = _nfc(s)[:max_len]
    if not t:
        raise ValueError("Message is empty")
    if "<" in t or ">" in t:
        raise ValueError("Message may not contain HTML (no < or > characters)")
    if _SUSPICIOUS.search(t):
        raise ValueError("Message contains disallowed content")
    # Remove most control chars; keep \n, \r, \t
    out = []
    for c in t:
        o = ord(c)
        if c in "\n\r\t":
            out.append(c)
        elif 32 <= o < 0x110000 and not (0xD800 <= o <= 0xDFFF):
            out.append(c)
        else:
            raise ValueError("Invalid characters in message")
    # Collapse pathological runs of newlines
    text = "".join(out)
    text = re.sub(r"\n{8,}", "\n\n\n\n\n\n\n", text)
    return text


def should_block_honeypot(website: Optional[str]) -> bool:
    return bool(website and str(website).strip())
