from typing import Annotated

from pydantic import BaseModel
from fastapi import UploadFile


class InputData(BaseModel):
    captured_image: Annotated[UploadFile, "image/jpg"]
    json_depth_data: Annotated[UploadFile, "application/json"]
    is_real: bool = True


class ResponseData(BaseModel):
    status_code: int
    file_id: str
    detail: str
