from pydantic import BaseModel
from typing import Optional, List, Literal
from datetime import datetime


class UpscalePhotoshootRequest(BaseModel):
    photoshoot_id:     str
    image_ids:         List[str]
    regeneration_type: Literal["upscale (2x)", "upscale (4x)"]


class RegeneratePhotoshootRequest(BaseModel):
    photoshoot_id: str
    image_ids:     Optional[List[str]] = []   # empty = regenerate all poses


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
    which_pose_option:            Literal["default", "custom", "prompt"]
    poses_ids:                    Optional[List[str]] = []
    poses_images:                 Optional[List[str]] = []
    poses_prompts:                Optional[List[str]] = []
    model_id:                     str
    lighting_style:               str
    ornaments:                    Optional[str]  = ""
    sku_id:                       Optional[str]  = ""
    regeneration_type:            Optional[str]  = ""
    regenerate_photoshoot_id:     Optional[str]  = ""
