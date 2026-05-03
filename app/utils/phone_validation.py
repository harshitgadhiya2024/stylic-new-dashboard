"""
International phone validation using Google's libphonenumber (``phonenumbers``).

- Validates numbering plan, length, and carrier/region rules for 200+ regions.
- Normalizes stored values to **E.164** (e.g. ``+14155552671``, ``+919876543210``).
- Numbers **without** a leading ``+`` are parsed as **national** format only when
  ``settings.PHONE_DEFAULT_REGION`` is set (ISO 3166-1 alpha-2, e.g. ``IN``, ``US``).
"""

from __future__ import annotations

from typing import Optional, Union

import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat

from app.config import settings


def _default_region() -> Optional[str]:
    r = (getattr(settings, "PHONE_DEFAULT_REGION", "") or "").strip().upper()
    if not r or len(r) != 2:
        return None
    if not r.isalpha():
        return None
    return r


def normalize_phone_to_e164(raw: str) -> str:
    """
    Strict validation: must be a possible **and** valid number for its region.

    Accepts:
    - International: leading ``+`` and country calling code (preferred).
    - National: digits/spaces/punctuation only if ``PHONE_DEFAULT_REGION`` is configured.

    Returns E.164. Raises ``ValueError`` with a short message for clients.
    """
    s = (raw or "").strip()
    if not s:
        raise ValueError("Phone number cannot be empty.")

    if s.startswith("00"):
        s = "+" + s[2:]

    region = _default_region()
    try:
        num = phonenumbers.parse(s, region)
    except NumberParseException as exc:
        hint = (
            " Use international format with country code (e.g. +1…, +91…)."
            if not region
            else (
                " Use international format (+country code) or a valid national number "
                f"for region {region}."
            )
        )
        raise ValueError("Could not parse phone number." + hint) from exc

    if not phonenumbers.is_possible_number(num):
        raise ValueError(
            "Phone number length or format is not possible for this country or type."
        )

    if not phonenumbers.is_valid_number(num):
        raise ValueError(
            "This phone number is not valid for its country (check digits and area/mobile prefix)."
        )

    return phonenumbers.format_number(num, PhoneNumberFormat.E164)


def normalize_phone_optional(raw: Optional[str]) -> Optional[str]:
    """``None`` / blank → ``None``; otherwise same as ``normalize_phone_to_e164``."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return normalize_phone_to_e164(s)


def normalize_phone_profile_field(raw: Optional[str]) -> Union[str, None]:
    """
    Profile PATCH semantics:
    - ``None`` → ``None`` (caller: field not in payload — do not change).
    - Blank string → ``""`` (explicit clear in DB).
    - Non-empty → E.164 string.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return ""
    return normalize_phone_to_e164(s)


def coerce_loose_phone_to_e164_or_empty(raw: Optional[str]) -> str:
    """
    For third-party tokens (e.g. Firebase): return E.164 when parseable, else ``""``
    without raising.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    try:
        return normalize_phone_to_e164(s)
    except ValueError:
        return ""
