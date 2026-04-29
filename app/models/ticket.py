"""User support ticket request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class CreateTicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_type: str = Field(..., min_length=1, max_length=200)
    descriptions: str = Field(..., min_length=1, max_length=20_000)
    images: List[str] = Field(default_factory=list, max_length=50)
    notes: str = Field(default="", max_length=5000)


class TicketRecord(BaseModel):
    ticket_id: str
    user_id: str
    ticket_type: str
    descriptions: str
    images: list[str]
    notes: str
    status: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
