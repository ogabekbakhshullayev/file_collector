from fastapi import APIRouter, HTTPException, File, UploadFile, Form

from models.prediction import ResponseData
import aiofiles
from typing import Annotated

import os

router = APIRouter()

facepay_data_dir = "/facepay-data"

@router.post("/rgb-ir-result-save", response_model=ResponseData)
async def predict(
    rgb_image: Annotated[UploadFile, File(...)],
    ir_image: Annotated[UploadFile, File(...)],
    result: str = Form(...),
    status: str = Form(...),
    file_id: str = Form(...),
):
    if not rgb_image or not ir_image or not result:
        raise HTTPException(status_code=404, detail="Files not found!")
    

    try:
        save_dir = f"{facepay_data_dir}/{status}/{file_id}"
        os.makedirs(save_dir, exist_ok=True)

        rgb_image_path = f"{save_dir}/rgb_image.jpeg"
        ir_image_path = f"{save_dir}/ir_image.jpeg"
        result_path = f"{save_dir}/result.json"

        if os.path.exists(rgb_image_path):
            os.remove(rgb_image_path)

        if os.path.exists(ir_image_path):
            os.remove(ir_image_path)

        if os.path.exists(result_path):
            os.remove(result_path)

        async with aiofiles.open(rgb_image_path, "wb") as out_file:
            content = await rgb_image.read()
            await out_file.write(content)

        async with aiofiles.open(ir_image_path, "wb") as out_file:
            content = await ir_image.read()
            await out_file.write(content)

        async with aiofiles.open(result_path, "w") as out_file:
            await out_file.write(result)


    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return ResponseData(
        status_code=200,
        file_id=file_id,
        detail="RGB, IR, and result files saved successfully!",
    )