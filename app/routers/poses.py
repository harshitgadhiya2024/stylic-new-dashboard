from fastapi import APIRouter, Depends

from app.database import get_poses_collection
from app.dependencies import get_current_user
from app.models.pose import PoseListItem

router = APIRouter(prefix="/api/v1/poses", tags=["Poses"])


@router.get(
    "/",
    response_model=list[PoseListItem],
    summary="Get All Poses",
    description=(
        "Returns all records from the poses_data collection as pose_id + image URL. "
        "`image` maps from `image_url` in the document. Secured — requires auth token."
    ),
)
async def get_all_poses(
    current_user: dict = Depends(get_current_user),
):
    col = get_poses_collection()
    cursor = col.find(
        {},
        {"_id": 0, "pose_id": 1, "image_url": 1},
    ).sort("pose_id", 1)

    docs = await cursor.to_list(length=None)
    return [
        PoseListItem(
            pose_id=str(doc.get("pose_id", "")),
            image=str(doc.get("image_url", "") or ""),
        )
        for doc in docs
    ]
