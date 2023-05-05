from fastapi import APIRouter, HTTPException, File, UploadFile

from models.prediction import ResponseData
import aiofiles
from typing import Annotated

from uuid import uuid4
import os

router = APIRouter()


@router.post("/file-save-request", response_model=ResponseData)
async def predict(
    captured_image: Annotated[UploadFile, "image/jpg"] = File(...),
    json_depth_data: Annotated[UploadFile, "application/json"] = File(...),
    is_real: bool = True,
):
    if not json_depth_data or not captured_image:
        raise HTTPException(status_code=404, detail="Files not found!")

    base_path = "real" if is_real else "spoof"
    file_id = str(uuid4())
    folder_name = f"/images/{base_path}/{file_id}"

    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        async with aiofiles.open(f"{folder_name}/json_depth_data.json", "wb") as out_file:
            content = await json_depth_data.read()
            await out_file.write(content)

        async with aiofiles.open(f"{folder_name}/captured_image.jpg", "wb") as out_file:
            content = await captured_image.read()
            await out_file.write(content)

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return ResponseData(
        status_code=200,
        file_id=file_id,
        detail="File saved successfully!",
    )
