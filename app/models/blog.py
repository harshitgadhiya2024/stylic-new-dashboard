"""
Blog content — marketing / CMS style posts stored in MongoDB.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BlogStatus = Literal["draft", "published", "archived"]


class FAQItem(BaseModel):
    """A single Frequently Asked Question entry."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(..., min_length=1, max_length=1000)
    answer: str = Field(..., min_length=1, max_length=5000)


class TableOfContentItem(BaseModel):
    """A single Table-of-Contents entry."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(..., min_length=1, max_length=500)
    anchor: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Anchor / fragment id used to deep-link inside the blog page.",
    )
    level: Optional[int] = Field(
        default=None,
        ge=1,
        le=6,
        description="Heading depth (1=H1 … 6=H6).",
    )


class CreateBlogRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    # ── Core (required) ────────────────────────────────────────────────
    blog_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Heading / H1 — also used as default for meta/OG/Twitter titles.",
    )
    blog_tagline: str = Field(..., min_length=1, max_length=2000)
    blog_type: str = Field(..., min_length=1, max_length=200)
    blog_hero_image: str = Field(..., min_length=1, max_length=4000)
    author_name: str = Field(..., min_length=1, max_length=300)
    blog_html_content: str = Field(..., min_length=1)

    # ── SEO — meta tags (optional) ─────────────────────────────────────
    meta_title: Optional[str] = Field(default=None, max_length=500)
    meta_description: Optional[str] = Field(default=None, max_length=2000)
    meta_keywords: Optional[str] = Field(
        default=None,
        max_length=1000,
        description='Comma-separated meta keywords (e.g. "Stylic AI, ...").',
    )

    # ── Open Graph (optional) ──────────────────────────────────────────
    og_title: Optional[str] = Field(default=None, max_length=500)
    og_description: Optional[str] = Field(default=None, max_length=2000)
    og_image: Optional[str] = Field(default=None, max_length=4000)
    og_image_alt: Optional[str] = Field(default=None, max_length=500)
    og_url: Optional[str] = Field(default=None, max_length=2000)

    # ── Twitter (optional) ─────────────────────────────────────────────
    twitter_title: Optional[str] = Field(default=None, max_length=500)
    twitter_description: Optional[str] = Field(default=None, max_length=2000)
    twitter_image: Optional[str] = Field(default=None, max_length=4000)
    twitter_image_alt: Optional[str] = Field(default=None, max_length=500)

    # ── Hero image alt (optional) ──────────────────────────────────────
    blog_hero_image_alt: Optional[str] = Field(default=None, max_length=500)

    # ── URLs (optional) ────────────────────────────────────────────────
    blog_url: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Public blog URL or slug.",
    )
    canonical_url: Optional[str] = Field(default=None, max_length=2000)

    # ── Schema.org JSON-LD (optional) ──────────────────────────────────
    schema_org: Optional[dict[str, Any]] = Field(
        default=None,
        description="JSON-LD schema.org object injected on the blog page.",
    )

    # ── Table of Contents (optional) ───────────────────────────────────
    table_of_contents: Optional[list[TableOfContentItem]] = Field(
        default=None,
        description="Ordered list of TOC entries shown on the blog page.",
    )

    # ── FAQs (optional) ────────────────────────────────────────────────
    faqs: Optional[list[FAQItem]] = Field(
        default=None,
        description="List of question/answer pairs rendered as the FAQ section.",
    )

    # ── Post date / time (optional, used for scheduling/display) ───────
    blog_post_date: Optional[date] = Field(
        default=None,
        description="ISO date (YYYY-MM-DD) for the published date.",
    )
    blog_post_time: Optional[time] = Field(
        default=None,
        description="ISO time (HH:MM[:SS]) for the published time.",
    )


class UpdateBlogRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    blog_id: str = Field(..., min_length=1, max_length=64)

    # ── Core ───────────────────────────────────────────────────────────
    blog_name: Optional[str] = Field(default=None, min_length=1, max_length=500)
    blog_tagline: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    blog_type: Optional[str] = Field(default=None, min_length=1, max_length=200)
    blog_hero_image: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    author_name: Optional[str] = Field(default=None, min_length=1, max_length=300)
    blog_html_content: Optional[str] = Field(default=None, min_length=1)
    status: Optional[BlogStatus] = None

    # ── SEO meta ───────────────────────────────────────────────────────
    meta_title: Optional[str] = Field(default=None, max_length=500)
    meta_description: Optional[str] = Field(default=None, max_length=2000)
    meta_keywords: Optional[str] = Field(default=None, max_length=1000)

    # ── Open Graph ─────────────────────────────────────────────────────
    og_title: Optional[str] = Field(default=None, max_length=500)
    og_description: Optional[str] = Field(default=None, max_length=2000)
    og_image: Optional[str] = Field(default=None, max_length=4000)
    og_image_alt: Optional[str] = Field(default=None, max_length=500)
    og_url: Optional[str] = Field(default=None, max_length=2000)

    # ── Twitter ────────────────────────────────────────────────────────
    twitter_title: Optional[str] = Field(default=None, max_length=500)
    twitter_description: Optional[str] = Field(default=None, max_length=2000)
    twitter_image: Optional[str] = Field(default=None, max_length=4000)
    twitter_image_alt: Optional[str] = Field(default=None, max_length=500)

    # ── Hero image alt ─────────────────────────────────────────────────
    blog_hero_image_alt: Optional[str] = Field(default=None, max_length=500)

    # ── URLs ───────────────────────────────────────────────────────────
    blog_url: Optional[str] = Field(default=None, max_length=2000)
    canonical_url: Optional[str] = Field(default=None, max_length=2000)

    # ── Schema.org JSON-LD ─────────────────────────────────────────────
    schema_org: Optional[dict[str, Any]] = None

    # ── TOC + FAQs ─────────────────────────────────────────────────────
    table_of_contents: Optional[list[TableOfContentItem]] = None
    faqs: Optional[list[FAQItem]] = None

    # ── Post date / time ───────────────────────────────────────────────
    blog_post_date: Optional[date] = None
    blog_post_time: Optional[time] = None

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
                self.meta_title is not None,
                self.meta_description is not None,
                self.meta_keywords is not None,
                self.og_title is not None,
                self.og_description is not None,
                self.og_image is not None,
                self.og_image_alt is not None,
                self.og_url is not None,
                self.twitter_title is not None,
                self.twitter_description is not None,
                self.twitter_image is not None,
                self.twitter_image_alt is not None,
                self.blog_hero_image_alt is not None,
                self.blog_url is not None,
                self.canonical_url is not None,
                self.schema_org is not None,
                self.table_of_contents is not None,
                self.faqs is not None,
                self.blog_post_date is not None,
                self.blog_post_time is not None,
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

    # SEO meta
    meta_title: Optional[str] = None
    meta_description: Optional[str] = None
    meta_keywords: Optional[str] = None

    # Open Graph
    og_title: Optional[str] = None
    og_description: Optional[str] = None
    og_image: Optional[str] = None
    og_image_alt: Optional[str] = None
    og_url: Optional[str] = None

    # Twitter
    twitter_title: Optional[str] = None
    twitter_description: Optional[str] = None
    twitter_image: Optional[str] = None
    twitter_image_alt: Optional[str] = None

    # Hero image alt
    blog_hero_image_alt: Optional[str] = None

    # URLs
    blog_url: Optional[str] = None
    canonical_url: Optional[str] = None

    # Schema / TOC / FAQs
    schema_org: Optional[dict[str, Any]] = None
    table_of_contents: Optional[list[TableOfContentItem]] = None
    faqs: Optional[list[FAQItem]] = None

    # Post date/time
    blog_post_date: Optional[date] = None
    blog_post_time: Optional[time] = None
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
