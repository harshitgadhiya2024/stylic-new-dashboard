"""Admin dashboard analytics API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.database import (
    get_backgrounds_collection,
    get_blogs_collection,
    get_credit_history_collection,
    get_model_faces_collection,
    get_photoshoots_collection,
    get_poses_collection,
    get_remove_background_collection,
    get_upscaling_collection,
    get_user_upscaled_collection,
    get_users_collection,
)
from app.dependencies import require_admin_roles

router = APIRouter(
    prefix="/api/v1/admins/analytics",
    tags=["Admin dashboard analytics"],
)


def _public_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k != "_id"}


async def _sum_field(collection, field_name: str) -> float:
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total": {"$sum": {"$ifNull": [f"${field_name}", 0]}},
            }
        }
    ]
    rows = await collection.aggregate(pipeline).to_list(length=1)
    if not rows:
        return 0.0
    return float(rows[0].get("total", 0) or 0)


@router.get(
    "/admin-dashboard-analytics",
    summary="Admin dashboard analytics",
)
async def get_admin_dashboard_analytics(
    _admin: dict = Depends(require_admin_roles("superadmin", "admin", "developer", "blogger")),
) -> dict[str, Any]:
    _ = _admin
    users_col = get_users_collection()
    credit_history_col = get_credit_history_collection()
    photoshoots_col = get_photoshoots_collection()
    upscaling_col = get_upscaling_collection()
    backgrounds_col = get_backgrounds_collection()
    model_faces_col = get_model_faces_collection()
    poses_col = get_poses_collection()
    remove_bg_col = get_remove_background_collection()
    user_upscaled_col = get_user_upscaled_collection()
    blogs_col = get_blogs_collection()

    total_users = await users_col.count_documents({})
    total_premium_users = await users_col.count_documents(
        {"plan": {"$in": ["silver", "gold", "platinum", "enterprise"]}}
    )
    users_credit_sum = await _sum_field(users_col, "credits")
    used_credit_sum = await _sum_field(credit_history_col, "credit")

    total_photoshoot = await photoshoots_col.count_documents({})
    total_failed = await photoshoots_col.count_documents(
        {"status": {"$in": ["failed", "processing", "queue"]}}
    )
    total_images = await upscaling_col.count_documents({})

    total_custom_background = await backgrounds_col.count_documents({"is_default": False})
    total_custom_models = await model_faces_col.count_documents({"is_default": False})
    total_custom_poses = await poses_col.count_documents({"is_default": False})
    total_background_change = await credit_history_col.count_documents(
        {"regeneration_type": "background_change"}
    )
    total_regenerated = await credit_history_col.count_documents(
        {"regeneration_type": "regenerate"}
    )
    total_color_change = await credit_history_col.count_documents(
        {"regeneration_type": "color_change"}
    )
    total_upscaled = await credit_history_col.count_documents(
        {"regeneration_type": {"$in": ["upscale (4x)", "upscale (2x)", "upscale (8x)"]}}
    )
    total_fabric_change = await credit_history_col.count_documents(
        {"regeneration_type": "fabric_change"}
    )
    total_branding = await credit_history_col.count_documents(
        {"regeneration_type": "branding"}
    )
    total_remove_background = await remove_bg_col.count_documents({})
    total_users_upscaled = await user_upscaled_col.count_documents({})

    latest_5_photoshoots = [
        _public_doc(d)
        async for d in photoshoots_col.find({}).sort("created_at", -1).limit(5)
    ]
    latest_5_images = [
        _public_doc(d)
        async for d in upscaling_col.find({}).sort("created_at", -1).limit(5)
    ]
    latest_5_blogs = [
        _public_doc(d)
        async for d in blogs_col.find({}).sort("created_at", -1).limit(5)
    ]

    return {
        "users_mapping": {
            "total_users": total_users,
            "total_premium_user": total_premium_users,
            "total_credits": round(users_credit_sum + used_credit_sum, 4),
            "total_used_credits": round(used_credit_sum, 4),
        },
        "photoshoot_mapping": {
            "total_photoshoot": total_photoshoot,
            "total_failed": total_failed,
            "total_images": total_images,
        },
        "feature_mapping": {
            "total_custom_background": total_custom_background,
            "total_custom_models": total_custom_models,
            "total_custom_poses": total_custom_poses,
            "total_background_change": total_background_change,
            "total_regerated": total_regenerated,
            "total_color_change": total_color_change,
            "total_upscaled": total_upscaled,
            "total_fabric_change": total_fabric_change,
            "total_branding": total_branding,
            "total_remove_background": total_remove_background,
            "total_users_upscaled": total_users_upscaled,
        },
        "latest_5_photoshoots": latest_5_photoshoots,
        "latest_5_images": latest_5_images,
        "latest_5_blogs": latest_5_blogs,
    }
