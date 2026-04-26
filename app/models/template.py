from typing import Literal, Optional

from pydantic import BaseModel, field_validator


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
    font_size:             str
    logo_size:             str

    @field_validator("font_size", mode="before")
    @classmethod
    def validate_font_size(cls, v: object) -> str:
        s = str(v).strip() if v is not None else ""
        if not s:
            raise ValueError("font_size is required")
        if not s.lower().endswith("px"):
            raise ValueError("font_size must be in px format (e.g., 16px)")
        num = s[:-2].strip()
        try:
            if float(num) <= 0:
                raise ValueError
        except Exception:
            raise ValueError("font_size must be in px format (e.g., 16px)")
        return f"{num}px"

    @field_validator("logo_size", mode="before")
    @classmethod
    def validate_logo_size(cls, v: object) -> str:
        s = str(v).strip() if v is not None else ""
        if not s:
            raise ValueError("logo_size is required")
        if not s.endswith("%"):
            raise ValueError("logo_size must be in % format (e.g., 25%)")
        num = s[:-1].strip()
        try:
            val = float(num)
            if val <= 0:
                raise ValueError
        except Exception:
            raise ValueError("logo_size must be in % format (e.g., 25%)")
        return f"{num}%"


class UpdateBrandingTemplateRequest(BaseModel):
    template_id:           str
    user_id:               str
    template_name:         Optional[str]         = None
    logo_image_url:        Optional[str]         = None
    serial_code_format:    Optional[str]         = None
    logo_position:         Optional[BrandingPosition] = None
    serial_code_position:  Optional[BrandingPosition] = None
    font_size:             Optional[str]         = None
    logo_size:             Optional[str]         = None
    is_active:             Optional[bool]        = None
    is_default:            Optional[bool]        = None

    @field_validator("font_size", mode="before")
    @classmethod
    def validate_font_size_optional(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            raise ValueError("font_size cannot be empty")
        if not s.lower().endswith("px"):
            raise ValueError("font_size must be in px format (e.g., 16px)")
        num = s[:-2].strip()
        try:
            if float(num) <= 0:
                raise ValueError
        except Exception:
            raise ValueError("font_size must be in px format (e.g., 16px)")
        return f"{num}px"

    @field_validator("logo_size", mode="before")
    @classmethod
    def validate_logo_size_optional(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            raise ValueError("logo_size cannot be empty")
        if not s.endswith("%"):
            raise ValueError("logo_size must be in % format (e.g., 25%)")
        num = s[:-1].strip()
        try:
            val = float(num)
            if val <= 0:
                raise ValueError
        except Exception:
            raise ValueError("logo_size must be in % format (e.g., 25%)")
        return f"{num}%"


class SoftDeleteBrandingTemplateRequest(BaseModel):
    template_id: str
    user_id:     str


class SetDefaultBrandingTemplateRequest(BaseModel):
    user_id:     str
    template_id: str
