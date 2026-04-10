from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Literal
from datetime import datetime


class UpscalePhotoshootRequest(BaseModel):
    photoshoot_id:     str
    image_ids:         List[str]
    regeneration_type: Literal["upscale (2x)", "upscale (4x)"]


class RegeneratePhotoshootRequest(BaseModel):
    photoshoot_id: str
    image_ids:     Optional[List[str]] = []   # empty = regenerate all poses


class DeletePhotoshootsRequest(BaseModel):
    photoshoot_ids: List[str]


class ResizeImageItem(BaseModel):
    image_id: str
    image:    str


class ResizePhotoshootRequest(BaseModel):
    photoshoot_id: str
    resize_list:   List[ResizeImageItem]


class BrandingPhotoshootRequest(BaseModel):
    photoshoot_id: str
    image_ids:     List[str]


class BackgroundChangeRequest(BaseModel):
    photoshoot_id: str
    background_id: str
    image_ids:     List[str]


class FabricChangeRequest(BaseModel):
    """API-1: Change fabric on a single garment image using Gemini."""
    photoshoot_id: str
    fabric:        str   # e.g. "cotton", "silk", "denim"


class FabricPhotoshootRequest(BaseModel):
    """API-2: Re-generate photoshoot with a new garment fabric via full SeedDream pipeline."""
    photoshoot_id:       str
    front_garment_image: str
    back_garment_image:  Optional[str] = ""
    image_ids:           List[str]


class TextureChangeRequest(BaseModel):
    """API-1: Change texture on a single garment image using Gemini."""
    photoshoot_id: str
    texture:       str   # e.g. "plain weave", "checked", "printed", "striped"


class TexturePhotoshootRequest(BaseModel):
    """API-2: Re-generate photoshoot with a new garment texture via full SeedDream pipeline."""
    photoshoot_id:       str
    front_garment_image: str
    back_garment_image:  Optional[str] = ""
    image_ids:           List[str]


class ColorChangeRequest(BaseModel):
    """API-1: Change color on a single garment image using Gemini."""
    photoshoot_id: str
    color_hex:     str   # e.g. "#fff28f", "#fe2a3e"


class ColorPhotoshootRequest(BaseModel):
    """API-2: Re-generate photoshoot with a new garment color via full SeedDream pipeline."""
    photoshoot_id:       str
    front_garment_image: str
    back_garment_image:  Optional[str] = ""
    image_ids:           List[str]


class CreatePhotoshootRequest(BaseModel):
    front_garment_image:          str
    back_garment_image:           Optional[str]  = ""
    ethnicity:                    str
    gender:                       str
    skin_tone:                    str
    age:                          str
    age_group:                    str
    weight:                       Optional[str]  = "regular"
    height:                       Optional[str]  = "regular"
    upper_garment_type:           Optional[str]  = ""
    upper_garment_specification:  Optional[str]  = ""
    lower_garment_type:           Optional[str]  = ""
    lower_garment_specification:  Optional[str]  = ""
    one_piece_garment_type:       Optional[str]  = ""
    one_piece_garment_specification: Optional[str] = ""
    fitting:                      Optional[str]  = "regular fit"
    background_id:                str
    poses_ids:                    List[str] = Field(
        ...,
        min_length=1,
        description="At least one pose_id from the poses collection (stored pose_prompt is used).",
    )
    model_id:                     str
    lighting_style:               str
    ornaments:                    Optional[str]  = ""
    sku_id:                       Optional[str]  = ""
    regeneration_type:            Optional[str]  = ""
    regenerate_photoshoot_id:     Optional[str]  = ""

    @field_validator("poses_ids", mode="before")
    @classmethod
    def _normalize_pose_ids(cls, v):
        if v is None:
            raise ValueError("poses_ids is required")
        if not isinstance(v, list):
            raise ValueError("poses_ids must be a list of strings")
        out = [str(x).strip() for x in v if x is not None and str(x).strip()]
        if not out:
            raise ValueError("poses_ids must contain at least one non-empty pose_id")
        return out


class PhotoshootBatchFieldMixin(BaseModel):
    """Shared optional fields for batch defaults and per-item overrides (merge: item wins)."""

    model_config = ConfigDict(extra="forbid")

    ethnicity: Optional[str] = None
    gender: Optional[str] = None
    skin_tone: Optional[str] = None
    age: Optional[str] = None
    age_group: Optional[str] = None
    weight: Optional[str] = None
    height: Optional[str] = None
    upper_garment_type: Optional[str] = None
    upper_garment_specification: Optional[str] = None
    lower_garment_type: Optional[str] = None
    lower_garment_specification: Optional[str] = None
    one_piece_garment_type: Optional[str] = None
    one_piece_garment_specification: Optional[str] = None
    fitting: Optional[str] = None
    background_id: Optional[str] = None
    poses_ids: Optional[List[str]] = None
    model_id: Optional[str] = None
    lighting_style: Optional[str] = None
    ornaments: Optional[str] = None
    front_garment_image: Optional[str] = None
    back_garment_image: Optional[str] = None
    sku_id: Optional[str] = None
    regeneration_type: Optional[str] = None
    regenerate_photoshoot_id: Optional[str] = None


class PhotoshootBatchDefaults(PhotoshootBatchFieldMixin):
    """Fields shared by every photoshoot in a batch. Omitted keys can be set per list item."""


class PhotoshootBatchListItem(BaseModel):
    """One photoshoot in a batch. Non-null fields override ``default_config`` for this row only."""

    model_config = ConfigDict(extra="ignore")

    ethnicity: Optional[str] = None
    gender: Optional[str] = None
    skin_tone: Optional[str] = None
    age: Optional[str] = None
    age_group: Optional[str] = None
    weight: Optional[str] = None
    height: Optional[str] = None
    upper_garment_type: Optional[str] = None
    upper_garment_specification: Optional[str] = None
    lower_garment_type: Optional[str] = None
    lower_garment_specification: Optional[str] = None
    one_piece_garment_type: Optional[str] = None
    one_piece_garment_specification: Optional[str] = None
    fitting: Optional[str] = None
    background_id: Optional[str] = None
    poses_ids: Optional[List[str]] = None
    model_id: Optional[str] = None
    lighting_style: Optional[str] = None
    ornaments: Optional[str] = None
    front_garment_image: Optional[str] = None
    back_garment_image: Optional[str] = None
    sku_id: Optional[str] = None
    regeneration_type: Optional[str] = None
    regenerate_photoshoot_id: Optional[str] = None


class CreateMultiplePhotoshootsRequest(BaseModel):
    default_config: PhotoshootBatchDefaults
    photoshoot_list_config: List[PhotoshootBatchListItem] = Field(
        ...,
        min_length=1,
        description="Each entry is merged onto default_config and processed as one independent photoshoot.",
    )
