from fastapi import APIRouter, HTTPException

from models.prediction import InputData, ResponseData

from uuid import uuid4
import os
import base64

router = APIRouter()


@router.post("/file-save-request", response_model=ResponseData)
async def predict(data_input: InputData):
    if not data_input:
        raise HTTPException(status_code=404, detail="'data_input' argument invalid!")
    file_id = str(uuid4())
    folder_name = f"/images/{file_id}"

    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(f"{folder_name}/first_image.jpg", "wb") as f:
            f.write(base64.b64decode(data_input.first_image))

        with open(f"{folder_name}/second_image.jpg", "wb") as f:
            f.write(base64.b64decode(data_input.second_image))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return ResponseData(
        status_code=200,
        file_id=file_id,
        detail="File saved successfully!",
    )
