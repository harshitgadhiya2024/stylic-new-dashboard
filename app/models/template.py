from typing import Literal, Optional

from pydantic import BaseModel


BrandingPosition = Literal[
    "top_left",
    "top_center",
    "top_right",
    "center_left",
    "center_center",
    "center_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
]


class CreateBrandingTemplateRequest(BaseModel):
    template_name:         str
    logo_image_url:        str
    serial_code_format:    str
    logo_position:         BrandingPosition
    serial_code_position:  BrandingPosition


class UpdateBrandingTemplateRequest(BaseModel):
    template_id:           str
    user_id:               str
    template_name:         Optional[str]         = None
    logo_image_url:        Optional[str]         = None
    serial_code_format:    Optional[str]         = None
    logo_position:         Optional[BrandingPosition] = None
    serial_code_position:  Optional[BrandingPosition] = None
    is_active:             Optional[bool]        = None
    is_default:            Optional[bool]        = None


class SoftDeleteBrandingTemplateRequest(BaseModel):
    template_id: str
    user_id:     str


class SetDefaultBrandingTemplateRequest(BaseModel):
    user_id:     str
    template_id: str
