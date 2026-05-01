"""
Blog content — marketing / CMS style posts stored in MongoDB.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BlogStatus = Literal["draft", "published", "archived"]


class CreateBlogRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    blog_name: str = Field(..., min_length=1, max_length=500)
    blog_tagline: str = Field(..., min_length=1, max_length=2000)
    blog_type: str = Field(..., min_length=1, max_length=200)
    blog_hero_image: str = Field(..., min_length=1, max_length=4000)
    author_name: str = Field(..., min_length=1, max_length=300)
    blog_html_content: str = Field(..., min_length=1)


class UpdateBlogRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    blog_id: str = Field(..., min_length=1, max_length=64)
    blog_name: Optional[str] = Field(default=None, min_length=1, max_length=500)
    blog_tagline: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    blog_type: Optional[str] = Field(default=None, min_length=1, max_length=200)
    blog_hero_image: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    author_name: Optional[str] = Field(default=None, min_length=1, max_length=300)
    blog_html_content: Optional[str] = Field(default=None, min_length=1)
    status: Optional[BlogStatus] = None

    @field_validator("blog_id", mode="after")
    @classmethod
    def _blog_id_strip(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _at_least_one_field(self) -> "UpdateBlogRequest":
        if not any(
            [
                self.blog_name is not None,
                self.blog_tagline is not None,
                self.blog_type is not None,
                self.blog_hero_image is not None,
                self.author_name is not None,
                self.blog_html_content is not None,
                self.status is not None,
            ]
        ):
            raise ValueError("At least one field besides blog_id must be provided to update.")
        return self


class UpdateBlogStatusRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    blog_id: str = Field(..., min_length=1, max_length=64)
    updated_status: BlogStatus = Field(
        ..., description="Target status: draft, published, or archived."
    )


class BlogRecord(BaseModel):
    """Response shape (aligned with DB — ISO datetimes in JSON)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    blog_id: str
    blog_name: str
    blog_tagline: str
    blog_type: str
    blog_hero_image: str
    author_name: str
    blog_html_content: str
    blog_post_date_and_time: Optional[datetime] = None
    status: str
    created_at: datetime
    updated_at: datetime


class AdminBlogListResponse(BaseModel):
    """Paginated admin blog list."""

    page: int
    limit: int
    total: int
    total_pages: int
    data: list[BlogRecord]
