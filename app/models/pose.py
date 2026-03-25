from pydantic import BaseModel


class PoseListItem(BaseModel):
    pose_id: str
    image:   str
