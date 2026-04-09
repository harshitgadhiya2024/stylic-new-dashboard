import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.database import get_model_faces_collection
from app.dependencies import get_current_user
from app.models.model_face import (
    CreateModelFaceRequest,
    CreateModelFaceWithAIRequest,
    DeleteModelFacesRequest,
    ModelFaceApiItem,
)
from app.services.ai_face_service import generate_and_upload_face_stream
from app.services.face_to_model_service import generate_model_face_from_reference_stream
from app.services.credit_service import check_sufficient_credits, deduct_credits_and_record

router = APIRouter(prefix="/api/v1/model-faces", tags=["Model Faces"])

# Stored in Mongo as snake_case; API accepts spaced or snake_case labels (any case).
_MODEL_CATEGORY_DB_VALUES = frozenset(
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

_MODEL_CATEGORY_FILTER_EXAMPLES = (
    "Baby, Kid Boy, Kid Girl, Young Boy, Young Girl, "
    "Adult Male, Adult Female, Senior Male, Senior Female"
)

_SSE_HEADERS = {
    "Cache-Control":    "no-cache",
    "X-Accel-Buffering": "no",
    "Connection":       "keep-alive",
}


def _clean_face(doc: dict) -> dict:
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


def _coerce_age_ethnicity_gender(doc: dict) -> tuple[Any, Any, Any]:
    """Prefer top-level fields; fall back to model_configuration."""
    cfg = doc.get("model_configuration") or {}
    age = doc.get("age")
    if age is None:
        age = cfg.get("age")
    ethnicity = doc.get("ethnicity")
    if ethnicity is None:
        ethnicity = cfg.get("ethnicity")
    gender = doc.get("gender")
    if gender is None:
        gender = cfg.get("gender")
    return age, ethnicity, gender


def _parse_age_for_filter(age_val: Any) -> Optional[int]:
    """Normalize stored age (int, float, or e.g. \"25 years\") to integer years, or None."""
    if age_val is None or isinstance(age_val, bool):
        return None
    if isinstance(age_val, int):
        return age_val
    if isinstance(age_val, float):
        return int(age_val)
    s = str(age_val).strip()
    if not s:
        return None
    m = re.match(r"^(\d+)", s)
    if m:
        return int(m.group(1))
    return None


def _doc_numeric_age(doc: dict) -> Optional[int]:
    age, _, _ = _coerce_age_ethnicity_gender(doc)
    return _parse_age_for_filter(age)


def _doc_ethnicity_normalized(doc: dict) -> Optional[str]:
    _, eth, _ = _coerce_age_ethnicity_gender(doc)
    if eth is None:
        return None
    s = str(eth).strip()
    return s.lower() if s else None


def _passes_age_ethnicity_filters(
    doc: dict,
    min_age: Optional[int],
    max_age: Optional[int],
    ethnicity_normalized: Optional[str],
) -> bool:
    if min_age is not None or max_age is not None:
        a = _doc_numeric_age(doc)
        if a is None:
            return False
        if min_age is not None and a < min_age:
            return False
        if max_age is not None and a > max_age:
            return False
    if ethnicity_normalized is not None:
        e = _doc_ethnicity_normalized(doc)
        if e is None:
            return False
        if e != ethnicity_normalized:
            return False
    return True


def serialize_model_face_response(doc: dict) -> dict:
    """
    Canonical API shape (testing.txt). Default faces: no user_id / is_favorite.
    Custom faces: includes user_id and is_favorite.
    """
    d = _clean_face(doc)
    cfg = d.get("model_configuration")
    if cfg is None:
        cfg = {}
    age, ethnicity, gender = _coerce_age_ethnicity_gender(d)
    fallback_ts = datetime.now(timezone.utc)
    created_at = d.get("created_at") or fallback_ts
    updated_at = d.get("updated_at") or fallback_ts
    out: dict = {
        "model_id":            d["model_id"],
        "model_name":          d.get("model_name") or "",
        "model_category":      d.get("model_category") or "",
        "model_configuration": cfg,
        "age":                 age,
        "ethnicity":           ethnicity,
        "gender":              gender,
        "tags":                d.get("tags") or [],
        "notes":               d.get("notes") or "",
        "model_used_count":    int(d.get("model_used_count", 0) or 0),
        "face_url":            d.get("face_url") or "",
        "favorite_list":       d.get("favorite_list") or [],
        "plan":                d.get("plan") or "silver",
        "is_default":          bool(d.get("is_default", False)),
        "is_active":           bool(d.get("is_active", True)),
        "created_at":          created_at,
        "updated_at":          updated_at,
    }
    if not out["is_default"]:
        out["is_favorite"] = bool(d.get("is_favorite", False))
        out["user_id"] = d.get("user_id")
    return out


def _jsonable_model_face_for_sse(doc: dict) -> dict:
    """Same as serialize_model_face_response but datetimes -> ISO strings for json.dumps."""
    data = serialize_model_face_response(doc)
    for key, val in list(data.items()):
        if isinstance(val, datetime):
            data[key] = val.isoformat()
    return data


def _normalize_model_category_filter(raw: Optional[str]) -> Optional[str]:
    """
    Map frontend labels (e.g. \"Adult Male\", \"Young Girl\") or snake_case to DB value.
    Returns None when ``raw`` is empty (no filter). Raises HTTPException 422 if unknown.
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    key = s.lower().replace(" ", "_").replace("-", "_")
    # Collapse accidental double underscores
    while "__" in key:
        key = key.replace("__", "_")
    if key in _MODEL_CATEGORY_DB_VALUES:
        return key
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=(
            f"Invalid model_category {raw!r}. "
            f"Use one of: {_MODEL_CATEGORY_FILTER_EXAMPLES} "
            "(or snake_case: baby, kid_boy, kid_girl, young_boy, young_girl, "
            "adult_male, adult_female, senior_male, senior_female)."
        ),
    )


def _sse(event: str, data: dict) -> str:
    """Format a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get(
    "/",
    summary="Get Model Faces",
    description=(
        "Returns a paginated list of model faces based on `type`. "
        "`type=default` — returns all platform default faces (is_default=True). "
        "`type=custom` — returns faces created by the current user (user_id match, is_active=True), "
        "sorted with favorites first then newest first. "
        "Optional filters: `is_favorite` (true/false), `model_category` "
        f"(e.g. {_MODEL_CATEGORY_FILTER_EXAMPLES} — matched to stored snake_case), "
        "`min_age` / `max_age` (inclusive, integer years; uses top-level or model_configuration.age), "
        "and `ethnicity` (case-insensitive exact match on top-level or model_configuration). "
        "Use `page` and `limit` to control pagination."
    ),
)
async def get_model_faces(
    type:  Literal["default", "custom"] = Query(..., description="'default' for platform faces, 'custom' for user-created faces"),
    page:  int = Query(default=1,  ge=1,        description="Page number (1-based)"),
    limit: int = Query(default=10, ge=1, le=100, description="Number of items per page"),
    is_favorite: Optional[bool] = Query(
        default=None,
        description="When set, return only favorites (true) or only non-favorites (false). Omit for no filter.",
    ),
    model_category: Optional[str] = Query(
        default=None,
        description=(
            f"Optional. One of: {_MODEL_CATEGORY_FILTER_EXAMPLES} "
            "(any case / spaces), or snake_case: baby, kid_boy, kid_girl, young_boy, "
            "young_girl, adult_male, adult_female, senior_male, senior_female."
        ),
    ),
    min_age: Optional[int] = Query(
        default=None,
        ge=0,
        le=150,
        description="Inclusive minimum age in years. Rows without a parseable numeric age are excluded when min_age or max_age is set.",
    ),
    max_age: Optional[int] = Query(
        default=None,
        ge=0,
        le=150,
        description="Inclusive maximum age in years.",
    ),
    ethnicity: Optional[str] = Query(
        default=None,
        description="Filter by ethnicity; case-insensitive exact match against stored value (top-level or inside model_configuration).",
    ),
    current_user: dict = Depends(get_current_user),
):
    if min_age is not None and max_age is not None and min_age > max_age:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="min_age cannot be greater than max_age.",
        )

    col = get_model_faces_collection()

    cat = _normalize_model_category_filter(model_category)

    eth_norm: Optional[str] = None
    if ethnicity is not None:
        e = ethnicity.strip()
        if e:
            eth_norm = e.lower()

    if type == "default":
        query: dict = {"is_default": True}
    else:
        query = {"user_id": current_user["user_id"], "is_active": True}

    if cat is not None:
        query["model_category"] = cat
    if is_favorite is not None:
        query["is_favorite"] = is_favorite

    if type == "custom":
        cursor = col.find(query).sort([("is_favorite", -1), ("created_at", -1)])
    else:
        cursor = col.find(query)
    docs = await cursor.to_list(length=None)

    docs = [
        d
        for d in docs
        if _passes_age_ethnicity_filters(d, min_age, max_age, eth_norm)
    ]

    total       = len(docs)
    skip        = (page - 1) * limit
    paged       = docs[skip: skip + limit]
    total_pages = (total + limit - 1) // limit if total else 1

    return {
        "type":        type,
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": total_pages,
        "filters":     {
            "is_favorite":    is_favorite,
            "model_category": cat,
            "min_age":        min_age,
            "max_age":        max_age,
            "ethnicity":      ethnicity.strip() if ethnicity and ethnicity.strip() else None,
        },
        "data":        [serialize_model_face_response(doc) for doc in paged],
    }


@router.post(
    "/",
    summary="Upload / Create a Model Face (Streaming)",
    description=(
        "Accepts a reference face photo URL. Streams real-time progress via SSE. "
        "Validates the image for a real human face (Gemini), generates a professional "
        "model portrait replicating the reference face (SeedDream), uploads to S3, "
        "saves to DB, and deducts 2.5 credits in the background. "
        "Secured — user_id is taken from the auth token. "
        "Response is `text/event-stream`. Final event `done` contains the full model face record."
    ),
)
async def create_model_face(
    body: CreateModelFaceRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    check_sufficient_credits(current_user)
    body.model_category = body.model_category.lower().replace(" ", "_")
    async def event_stream() -> AsyncGenerator[str, None]:
        generated_face_url: str | None = None

        try:
            async for step, message, face_url in generate_model_face_from_reference_stream(
                body.reference_face_url,
                body.model_category,
            ):
                if step == "done":
                    generated_face_url = face_url
                    yield _sse("storing_db", {"step": "storing_db", "message": "Storing in database"})

                    now = datetime.now(timezone.utc)
                    mid = str(uuid.uuid4())
                    doc_db = {
                        "model_id":            mid,
                        "user_id":             current_user["user_id"],
                        "model_name":          body.model_name,
                        "model_category":      body.model_category,
                        "model_configuration": {},
                        "age":                 None,
                        "ethnicity":           None,
                        "gender":              None,
                        "tags":                body.tags or [],
                        "notes":               body.notes or "",
                        "model_used_count":    0,
                        "face_url":            generated_face_url,
                        "favorite_list":       [],
                        "plan":                "silver",
                        "reference_face_url":  body.reference_face_url,
                        "is_default":          False,
                        "is_active":           True,
                        "is_favorite":         False,
                        "created_at":          now,
                        "updated_at":          now,
                    }
                    col = get_model_faces_collection()
                    await col.insert_one(doc_db)
                    saved = await col.find_one({"model_id": mid})
                    if saved:
                        payload = _jsonable_model_face_for_sse(saved)
                    else:
                        payload = _jsonable_model_face_for_sse(doc_db)
                    yield _sse(
                        "done",
                        {
                            "step":    "done",
                            "message": "Face generation complete",
                            "data":    payload,
                        },
                    )
                else:
                    yield _sse(step, {"step": step, "message": message})
        except HTTPException as exc:
            yield _sse("error", {"step": "error", "message": exc.detail})
            return
        except Exception as exc:
            yield _sse("error", {"step": "error", "message": str(exc)})
            return

        if generated_face_url:
            background_tasks.add_task(
                deduct_credits_and_record,
                user=current_user,
                feature_name="face_generation",
                generated_face_url=generated_face_url,
                notes=body.notes or "",
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post(
    "/generate-with-ai",
    summary="Create Model Face Using AI (Streaming)",
    description=(
        "Streams real-time progress via SSE. Generates a passport-style portrait via kie.ai "
        "nano-banana-pro using the same prompt pattern as scripts/generate_model_faces.py "
        "(face configuration as JSON + headshot instructions). Optional fields use "
        "category-appropriate defaults; uploads to S3, saves to DB, and deducts 2.5 credits "
        "in the background. beard_length and beard_color only apply when model_category is "
        "'adult_male'. Response is `text/event-stream`. Final event `done` contains the full "
        "model face record."
    ),
)
async def create_model_face_with_ai(
    body: CreateModelFaceWithAIRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    check_sufficient_credits(current_user)
    body.model_category = body.model_category.lower().replace(" ", "_")
    overrides = {}
    if body.face_configurations:
        overrides = {
            k: v
            for k, v in body.face_configurations.model_dump().items()
            if v is not None
        }

    async def event_stream() -> AsyncGenerator[str, None]:
        generated_face_url: str | None = None
        final_config: dict | None = None

        try:
            async for step, message, face_url, config in generate_and_upload_face_stream(
                body.model_category,
                overrides,
            ):
                if step == "done":
                    generated_face_url = face_url
                    final_config       = config or {}
                    yield _sse("storing_db", {"step": "storing_db", "message": "Storing in database"})

                    now = datetime.now(timezone.utc)
                    mid = str(uuid.uuid4())
                    age = final_config.get("age")
                    ethnicity = final_config.get("ethnicity")
                    gender = final_config.get("gender")
                    doc_db = {
                        "model_id":            mid,
                        "user_id":             current_user["user_id"],
                        "model_name":          body.model_name,
                        "model_category":      body.model_category,
                        "model_configuration": final_config,
                        "age":                 age,
                        "ethnicity":           ethnicity,
                        "gender":              gender,
                        "tags":                body.tags or [],
                        "notes":               body.notes or "",
                        "model_used_count":    0,
                        "face_url":            generated_face_url,
                        "favorite_list":       [],
                        "plan":                "silver",
                        "reference_face_url":  None,
                        "is_default":          False,
                        "is_active":           True,
                        "is_favorite":         False,
                        "created_at":          now,
                        "updated_at":          now,
                    }
                    col = get_model_faces_collection()
                    await col.insert_one(doc_db)
                    saved = await col.find_one({"model_id": mid})
                    if saved:
                        payload = _jsonable_model_face_for_sse(saved)
                    else:
                        payload = _jsonable_model_face_for_sse(doc_db)
                    yield _sse(
                        "done",
                        {
                            "step":    "done",
                            "message": "Face generation complete",
                            "data":    payload,
                        },
                    )
                else:
                    yield _sse(step, {"step": step, "message": message})
        except HTTPException as exc:
            yield _sse("error", {"step": "error", "message": exc.detail})
            return
        except Exception as exc:
            yield _sse("error", {"step": "error", "message": str(exc)})
            return

        if generated_face_url:
            background_tasks.add_task(
                deduct_credits_and_record,
                user=current_user,
                feature_name="face_generation",
                generated_face_url=generated_face_url,
                notes=body.notes or "",
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.patch(
    "/{model_id}/toggle-favorite",
    response_model=ModelFaceApiItem,
    summary="Toggle Favorite",
    description="Switch is_favorite between true and false for a model face. Secured — only the owner can toggle.",
)
async def toggle_favorite(
    model_id: str,
    current_user: dict = Depends(get_current_user),
):
    col = get_model_faces_collection()

    doc = await col.find_one({"model_id": model_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model face not found.",
        )

    if doc.get("is_default", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Default model faces cannot be marked as favorite.",
        )

    if doc.get("user_id") != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this model face.",
        )

    new_value = not doc.get("is_favorite", False)
    now = datetime.now(timezone.utc)

    await col.update_one(
        {"model_id": model_id},
        {"$set": {"is_favorite": new_value, "updated_at": now}},
    )

    doc["is_favorite"] = new_value
    doc["updated_at"]  = now
    return ModelFaceApiItem(**serialize_model_face_response(doc))


@router.delete(
    "/bulk-delete",
    summary="Delete Multiple Model Faces",
    description=(
        "Soft-deletes multiple model faces by setting is_active=False. "
        "Only faces owned by the current user (user_id match) and not default (is_default=False) "
        "are deleted. Default faces are silently skipped. "
        "Returns counts of deleted and skipped faces."
    ),
)
async def delete_model_faces_bulk(
    body: DeleteModelFacesRequest,
    current_user: dict = Depends(get_current_user),
):
    if not body.model_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="model_ids must not be empty.",
        )

    col     = get_model_faces_collection()
    user_id = current_user["user_id"]

    result = await col.update_many(
        {
            "model_id":  {"$in": body.model_ids},
            "user_id":   user_id,
            "is_default": False,
            "is_active":  True,
        },
        {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
    )

    deleted_count = result.modified_count
    skipped_count = len(body.model_ids) - deleted_count

    return {
        "message":       f"{deleted_count} model face(s) deleted successfully.",
        "deleted_count": deleted_count,
        "skipped_count": skipped_count,
    }


@router.delete(
    "/{model_id}",
    summary="Delete Model Face",
    description="Soft-delete a model face by setting is_active=False. Secured — only the owner can delete.",
)
async def delete_model_face(
    model_id: str,
    current_user: dict = Depends(get_current_user),
):
    col = get_model_faces_collection()

    doc = await col.find_one({"model_id": model_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model face not found.",
        )

    if doc.get("is_default", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Default model faces cannot be deleted.",
        )

    if doc.get("user_id") != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this model face.",
        )

    if not doc.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model face is already deleted.",
        )

    await col.update_one(
        {"model_id": model_id},
        {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
    )

    return {"message": "Model face deleted successfully.", "model_id": model_id}
