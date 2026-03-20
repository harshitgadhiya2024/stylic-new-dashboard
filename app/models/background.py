from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class DeleteBackgroundsRequest(BaseModel):
    background_ids: list[str]


class CreateBackgroundRequest(BaseModel):
    background_name: str
    background_url:  str
    tags:            Optional[List[str]] = []
    notes:           Optional[str]       = ""


class CreateBackgroundWithAIRequest(BaseModel):
    background_name:          str
    background_configuration: str
    tags:                     Optional[List[str]] = []
    notes:                    Optional[str]       = ""


class BackgroundSchema(BaseModel):
    background_id:   str
    user_id:         Optional[str] = None
    background_type: str
    background_name: str
    background_url:  str
    count:           int           = 0
    tags:            List[str]     = []
    notes:           str           = ""
    is_default:      bool          = False
    is_active:       bool          = True
    created_at:      datetime
    updated_at:      datetime
