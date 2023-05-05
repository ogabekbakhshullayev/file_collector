from typing import Any

from pydantic import BaseModel


class InputData(BaseModel):
    first_image: str
    second_image: str


class ResponseData(BaseModel):
    status_code: int
    file_id: str
    detail: str
