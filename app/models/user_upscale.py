from typing import Literal

from pydantic import BaseModel, Field, field_validator


class UserUpscaleImageRequest(BaseModel):
    input_image_public_url: str = Field(..., min_length=1)
    upscale_factor: Literal[2, 4, 8]

    @field_validator("input_image_public_url", mode="before")
    @classmethod
    def strip_url(cls, v: object) -> str:
        s = str(v).strip() if v is not None else ""
        if not s:
            raise ValueError("input_image_public_url is required")
        return s


class UserUpscaleIdsRequest(BaseModel):
    upscale_ids: list[str] = Field(..., min_length=1)


class UserUpscaleIdRequest(BaseModel):
    upscale_id: str = Field(..., min_length=1)
