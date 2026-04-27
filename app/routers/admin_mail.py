"""Admin mail templates and send history — requires ``Authorization: Bearer`` admin access token."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.database import get_mail_sends_collection, get_mail_templates_collection
from app.dependencies import get_current_admin
from app.models.admin_mail import (
    MailSendRequest,
    MailTemplateCreateRequest,
    mail_send_public,
    mail_template_public,
)
from app.services.email_service import send_message_with_custom_from
from app.services.mail_template_service import render_template_string

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admins/mail", tags=["Admin mail"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


@router.post("/templates", status_code=status.HTTP_201_CREATED)
async def create_mail_template(
    body: MailTemplateCreateRequest,
    admin: dict = Depends(get_current_admin),
) -> dict[str, Any]:
    col = get_mail_templates_collection()
    mail_template_id = str(uuid.uuid4())
    ts = _now()
    doc = {
        "mail_template_id":  mail_template_id,
        "admin_id":          admin["admin_id"],
        "template_name":     body.template_name,
        "template_type":     body.template_type,
        "subject":           body.subject,
        "template_format":   body.template_format,
        "template_content":  body.template_content,
        "dynamic_variables": body.dynamic_variables,
        "created_at":        ts,
        "updated_at":        ts,
    }
    await col.insert_one(doc)
    return mail_template_public(doc)


@router.get("/templates")
async def list_mail_templates(
    admin: dict = Depends(get_current_admin),
) -> list[dict[str, Any]]:
    col = get_mail_templates_collection()
    cur = col.find({"admin_id": admin["admin_id"]}).sort("created_at", -1)
    return [mail_template_public(d) async for d in cur]


@router.get("/templates/{mail_template_id}")
async def get_mail_template(
    mail_template_id: str,
    admin: dict = Depends(get_current_admin),
) -> dict[str, Any]:
    col = get_mail_templates_collection()
    doc = await col.find_one(
        {"mail_template_id": mail_template_id, "admin_id": admin["admin_id"]},
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mail template not found.")
    return mail_template_public(doc)


@router.post("/send", status_code=status.HTTP_201_CREATED)
async def send_mail(
    body: MailSendRequest,
    admin: dict = Depends(get_current_admin),
) -> dict[str, Any]:
    tcol = get_mail_templates_collection()
    template = await tcol.find_one(
        {"mail_template_id": body.mail_template_id, "admin_id": admin["admin_id"]},
    )
    if not template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mail template not found.")

    fmt = template.get("template_format") or "html"
    is_html = fmt == "html"
    values = body.dynamic_variable_value or {}
    subj = render_template_string(
        str(template.get("subject") or ""),
        values,
        escape_for_html=False,
    )
    body_text = render_template_string(
        str(template.get("template_content") or ""),
        values,
        escape_for_html=is_html,
    )

    status_mapping: list[dict[str, Any]] = []
    for raw in body.receiver_mail_lists:
        to_addr = str(raw).strip().lower()
        try:
            await send_message_with_custom_from(
                to_email=to_addr,
                subject=subj,
                body=body_text,
                from_email=str(body.sender_mail),
                is_html=is_html,
            )
            status_mapping.append({"receiver_email": to_addr, "status": "sent"})
        except Exception as exc:
            logger.exception("send_mail failed for %s", to_addr)
            status_mapping.append(
                {
                    "receiver_email": to_addr,
                    "status":         "failed",
                    "error":          str(exc),
                },
            )

    mail_sender_id = str(uuid.uuid4())
    ts = _now()
    doc = {
        "mail_sender_id":            mail_sender_id,
        "admin_id":                  admin["admin_id"],
        "mail_template_id":          body.mail_template_id,
        "dynamic_variable_value":    dict(body.dynamic_variable_value or {}),
        "sender_mail":               str(body.sender_mail),
        "receiver_email_list":       [str(x).strip().lower() for x in body.receiver_mail_lists],
        "status_mapping":            status_mapping,
        "created_at":                ts,
        "updated_at":                ts,
    }
    scol = get_mail_sends_collection()
    await scol.insert_one(doc)
    return mail_send_public(doc)


@router.get("/sends")
async def list_mail_sends(
    admin: dict = Depends(get_current_admin),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
) -> list[dict[str, Any]]:
    col = get_mail_sends_collection()
    cur = (
        col.find({"admin_id": admin["admin_id"]})
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )
    return [mail_send_public(d) async for d in cur]


@router.get("/sends/{mail_sender_id}")
async def get_mail_send(
    mail_sender_id: str,
    admin: dict = Depends(get_current_admin),
) -> dict[str, Any]:
    col = get_mail_sends_collection()
    doc = await col.find_one(
        {"mail_sender_id": mail_sender_id, "admin_id": admin["admin_id"]},
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mail send record not found.")
    return mail_send_public(doc)
