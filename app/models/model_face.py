from typing import Any, Optional

from pydantic import BaseModel, ConfigDict
from datetime import datetime


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


class ModelFaceApiItem(BaseModel):
    """
    Canonical model-face shape for API responses (matches platform import format).
    Default faces omit user_id and is_favorite; custom faces include them.
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
