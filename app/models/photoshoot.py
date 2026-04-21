from pydantic import AliasChoices, BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Literal
from datetime import datetime


class UpscalePhotoshootRequest(BaseModel):
    photoshoot_id:     str
    image_ids:         List[str]
    regeneration_type: Literal["upscale (4x)", "upscale (8x)"]


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
    """API-1: Change upper vs lower garment fabric/material (kie.ai)."""
    photoshoot_id: str
    upper_fabric:  str = Field(..., min_length=1, description="Material for upper garment(s), e.g. silk, linen")
    lower_fabric:  str = Field(..., min_length=1, description="Material for lower garment(s), e.g. denim, wool")

    @field_validator("upper_fabric", "lower_fabric", mode="before")
    @classmethod
    def strip_fabric(cls, v: object) -> str:
        s = str(v).strip() if v is not None else ""
        if not s:
            raise ValueError("fabric description is required")
        return s


class FabricPhotoshootRequest(BaseModel):
    """API-2: Re-generate photoshoot with a new garment fabric via full SeedDream pipeline."""
    photoshoot_id:       str
    front_garment_image: str
    back_garment_image:  Optional[str] = ""
    image_ids:           List[str]


class TextureChangeRequest(BaseModel):
    """API-1: Change upper vs lower garment surface texture/pattern (kie.ai)."""
    photoshoot_id:  str
    upper_texture:  str = Field(..., min_length=1, description="Texture/pattern for upper garment(s), e.g. herringbone weave")
    lower_texture:  str = Field(..., min_length=1, description="Texture/pattern for lower garment(s), e.g. fine pinstripe")

    @field_validator("upper_texture", "lower_texture", mode="before")
    @classmethod
    def strip_texture(cls, v: object) -> str:
        s = str(v).strip() if v is not None else ""
        if not s:
            raise ValueError("texture description is required")
        return s


class TexturePhotoshootRequest(BaseModel):
    """API-2: Re-generate photoshoot with a new garment texture via full SeedDream pipeline."""
    photoshoot_id:       str
    front_garment_image: str
    back_garment_image:  Optional[str] = ""
    image_ids:           List[str]


class ColorChangeRequest(BaseModel):
    """API-1: Change upper vs lower garment colors on flat-lay / reference garment images (kie.ai)."""
    photoshoot_id:     str
    upper_color_hex:   str = Field(..., description="Target color for upper garment(s), e.g. #fe2a3e")
    lower_color_hex:   str = Field(..., description="Target color for lower garment(s), e.g. #112233")

    @field_validator("upper_color_hex", "lower_color_hex", mode="before")
    @classmethod
    def normalize_hex(cls, v: object) -> str:
        if v is None or (isinstance(v, str) and not str(v).strip()):
            raise ValueError("color hex is required")
        s = str(v).strip()
        if not s.startswith("#"):
            s = "#" + s
        body = s[1:]
        if len(body) == 3:
            body = "".join(c * 2 for c in body)
            s = "#" + body
            body = s[1:]
        if len(body) != 6 or any(c not in "0123456789abcdefABCDEF" for c in body):
            raise ValueError("color hex must be #RRGGBB (6 hex digits)")
        return s.lower()


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
    weight: Optional[str] = Field(
        default="regular",
        validation_alias=AliasChoices("weight", "body_weight"),
    )
    height: Optional[str] = Field(
        default="regular",
        validation_alias=AliasChoices("height", "body_height"),
    )
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
