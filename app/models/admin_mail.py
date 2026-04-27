"""Admin mail template & send models."""

from __future__ import annotations

from typing import Any, List, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator


class MailTemplateCreateRequest(BaseModel):
    template_name:       str = Field(..., min_length=1, max_length=300)
    template_type:       str = Field(..., min_length=1, max_length=100)
    subject:             str = Field(..., min_length=1, max_length=500)
    template_format:     Literal["html", "text"] = Field(
        "html",
        description="Send as HTML or plain text; same key stored in the database.",
    )
    template_content:    str = Field(
        ...,
        min_length=1,
        description="Body with placeholders such as {{name}} or {name} for dynamic_variables.",
    )
    dynamic_variables:  List[str] = Field(
        default_factory=list,
        description="Declared variable names for validation/documentation.",
    )

    @field_validator("dynamic_variables", mode="before")
    @classmethod
    def norm_vars(cls, v) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("dynamic_variables must be a list of strings")
        out: list[str] = []
        for x in v:
            s = str(x).strip()
            if s and s not in out:
                out.append(s)
        return out


class MailSendRequest(BaseModel):
    mail_template_id:        str = Field(..., min_length=1)
    dynamic_variable_value:  dict[str, Any] = Field(
        default_factory=dict,
        description="Map of variable name → value for template substitution",
    )
    sender_mail:             EmailStr
    receiver_mail_lists:     List[EmailStr] = Field(..., min_length=1)

    @field_validator("receiver_mail_lists", mode="before")
    @classmethod
    def dedupe_emails(cls, v) -> list:
        if not isinstance(v, list) or not v:
            raise ValueError("receiver_mail_lists must be a non-empty list of emails")
        seen: set[str] = set()
        out: list = []
        for x in v:
            e = str(x).strip().lower()
            if e and e not in seen:
                seen.add(e)
                out.append(e)
        return out


def mail_template_public(doc: dict) -> dict:
    d = dict(doc)
    d.pop("_id", None)
    return d


def mail_send_public(doc: dict) -> dict:
    d = dict(doc)
    d.pop("_id", None)
    return d
