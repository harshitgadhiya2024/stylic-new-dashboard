from pydantic import BaseModel, Field, field_validator


class RemoveBackgroundCreateRequest(BaseModel):
    input_image_public_url: str = Field(..., min_length=1, description="Public HTTP(S) URL of the image to process")

    @field_validator("input_image_public_url", mode="before")
    @classmethod
    def strip_url(cls, v: object) -> str:
        s = str(v).strip() if v is not None else ""
        if not s:
            raise ValueError("input_image_public_url is required")
        return s


class RemoveBackgroundIdsRequest(BaseModel):
    remove_background_ids: list[str] = Field(..., min_length=1)


class RemoveBackgroundIdRequest(BaseModel):
    remove_background_id: str = Field(..., min_length=1)

