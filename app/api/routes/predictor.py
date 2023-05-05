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

    folder_name = f"/app/images/{str(uuid4())}"
    print(folder_name)

    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(f"{folder_name}/image_1.jpg", "wb") as f:
            f.write(base64.b64decode(data_input.first_image))

        with open(f"{folder_name}/image_2.jpg", "wb") as f:
            f.write(base64.b64decode(data_input.second_image))

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return ResponseData(
        status_code=200,
        file_id=folder_name,
        detail="File saved successfully!",
    )
