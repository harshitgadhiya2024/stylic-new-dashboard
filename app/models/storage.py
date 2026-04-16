"""Request/response models for object storage APIs."""

from pydantic import BaseModel, Field


class DeleteR2ByPublicUrlRequest(BaseModel):
    """Body for deleting one object from R2 using its public URL."""

    public_url: str = Field(
        ...,
        min_length=12,
        description="Full HTTPS public URL of the file (must match configured R2_PUBLIC_URL).",
        examples=["https://cdn.example.com/users/your-user-id/photo.jpg"],
    )


class DeleteR2ByPublicUrlResponse(BaseModel):
    success: bool = True
    message: str = "Object deleted."
    object_key: str | None = Field(None, description="Object key that was removed from the bucket.")
