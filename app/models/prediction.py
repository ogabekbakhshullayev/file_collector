from fastapi import UploadFile, File
from pydantic import BaseModel
from typing import Annotated, Any


class FileCollectorInput(BaseModel):
    captured_image: Annotated[UploadFile, "image/jpg"] = (File(...),)
    json_depth_data: Annotated[UploadFile, "application/json"] = (File(...),)
    is_real: bool = True


class ResponseData(BaseModel):
    status_code: int
    file_id: str
    detail: str


class NearFarFacemeshSaveRequest(BaseModel):
    near_image: Annotated[UploadFile, "image/jpg"] = File(...)
    far_image: Annotated[UploadFile, "image/jpg"] = File(...)
    facemesh: Annotated[UploadFile, "application/json"] = File(...)
    file_id: str