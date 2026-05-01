from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime

_ALLOWED_ADMIN_MODEL_CATEGORIES = frozenset(
    {
        "baby",
        "kid_boy",
        "kid_girl",
        "young_boy",
        "young_girl",
        "adult_male",
        "adult_female",
        "senior_male",
        "senior_female",
    }
)


def normalize_model_category_value(value) -> str:
    """Map labels or snake_case to stored ``model_category``."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("model_category is required")
    key = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in key:
        key = key.replace("__", "_")
    if key in _ALLOWED_ADMIN_MODEL_CATEGORIES:
        return key
    raise ValueError(
        "model_category must be one of: baby, kid_boy, kid_girl, young_boy, young_girl, "
        "adult_male, adult_female, senior_male, senior_female "
        "(any spacing or casing)."
    )


class FaceConfiguration(BaseModel):
    face_shape:       Optional[str] = None
    jawline_type:     Optional[str] = None
    cheekbone_height: Optional[str] = None
    face_skin_tone:   Optional[str] = None
    skin_undertone:   Optional[str] = None
    hair_color:       Optional[str] = None
    hair_length:      Optional[str] = None
    hair_style:       Optional[str] = None
    eye_shape:        Optional[str] = None
    eye_color:        Optional[str] = None
    nose_shape:       Optional[str] = None
    lip_shape:        Optional[str] = None
    eyebrow_shape:    Optional[str] = None
    # beard fields — only applied when model_category is adult_male
    beard_length:     Optional[str] = None
    beard_color:      Optional[str] = None
    age:              Optional[str] = None
    ethnicity:        Optional[str] = None
    gender:           Optional[str] = None


class DeleteModelFacesRequest(BaseModel):
    model_ids: list[str]


class CreateModelFaceRequest(BaseModel):
    model_name: str
    model_category: str
    reference_face_url: str
    tags: Optional[list[str]] = []
    notes: Optional[str] = ""


class CreateModelFaceWithAIRequest(BaseModel):
    model_name: str
    model_category: str
    face_configurations: Optional[FaceConfiguration] = None
    tags: Optional[list[str]] = []
    notes: Optional[str] = ""


class ModelFaceSchema(BaseModel):
    """Legacy schema — prefer ModelFaceApiItem for API responses."""

    model_id: str
    user_id: Optional[str] = None
    model_name: str
    model_category: str
    model_configuration: dict = {}
    tags: list[str] = []
    notes: str = ""
    model_used_count: int = 0
    face_url: str
    reference_face_url: Optional[str] = None
    is_default: bool = False
    is_active: bool = True
    is_favorite: bool = False
    created_at: datetime
    updated_at: datetime


class AdminCreateDefaultModelFaceRequest(BaseModel):
    """Admin create platform default model face (see testing.txt)."""

    model_name: str = Field(..., min_length=1)
    model_category: str = Field(...)
    age: Union[int, str]
    ethnicity: str = Field(..., min_length=1)
    gender: str = Field(..., min_length=1)
    face_url: str = Field(..., min_length=1)

    @field_validator("model_name", "ethnicity", "gender", "face_url", mode="before")
    @classmethod
    def _strip_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("model_category", mode="before")
    @classmethod
    def _v_category(cls, v):
        return normalize_model_category_value(v)


class AdminUpdateModelFaceRequest(BaseModel):
    """Admin partial update for any model face."""

    model_name: Optional[str] = Field(None, min_length=1)
    model_category: Optional[str] = None
    age: Optional[Union[int, str]] = None
    ethnicity: Optional[str] = Field(None, min_length=1)
    gender: Optional[str] = Field(None, min_length=1)
    face_url: Optional[str] = Field(None, min_length=1)
    tags: Optional[list[str]] = None
    notes: Optional[str] = None
    model_used_count: Optional[int] = Field(None, ge=0)
    is_active: Optional[bool] = None

    @field_validator("model_name", "ethnicity", "gender", "face_url", mode="before")
    @classmethod
    def _strip_optional(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("model_category", mode="before")
    @classmethod
    def _v_cat_opt(cls, v):
        if v is None:
            return None
        return normalize_model_category_value(v)


class ModelFaceApiItem(BaseModel):
    """
    Canonical model-face shape for API responses (matches platform import format).
    Custom faces include user_id and document is_favorite.
    Default faces omit user_id; is_favorite is set when the viewer has bookmarked via favorite_list.
    """

    model_config = ConfigDict(extra="ignore")

    model_id: str
    model_name: str
    model_category: str
    model_configuration: dict = {}
    age: Optional[Any] = None
    ethnicity: Optional[str] = None
    gender: Optional[str] = None
    tags: list[str] = []
    notes: str = ""
    model_used_count: int = 0
    face_url: str
    favorite_list: list[Any] = []
    plan: str = "silver"
    is_default: bool = False
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    user_id: Optional[str] = None
    is_favorite: Optional[bool] = None
