from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Body

from models.prediction import SiliconSaveResponse
import aiofiles
from typing import Annotated
import aiohttp
from uuid import uuid4
import os
import base64

router = APIRouter()

vision_url = "http://213.230.069.228:31888/api/v1/detect-liveness"


def base64_2_bytestr(image_str: str):
    if "base64," in image_str:
        image_str = image_str.split("base64,")[1]
    image_str = base64.urlsafe_b64decode(fix_b64_padding(image_str))
    return image_str

def fix_b64_padding(b64_string):
    return f"{b64_string}{'=' * (len(b64_string) % 4)}"


@router.post("/3d-images", response_model=SiliconSaveResponse)
async def predict(
    image_b64: str = Body(...),
    file_id: str = Body(...)
):
    if not image_b64:
        raise HTTPException(status_code=404, detail="Base64 image not found!")
    
    payload = {
        'single_face_only': True,
        'color_image_only': True,
        'blink_threshold': 0.5,
        'head_rotation': True,
        'edge_detection': True,
        'blurriness': {'pixel_based': True},
        'model': 'fasnet-bbnet_v14-bbnet_full_v4-yolo_liveness_v1',
        'obj_id': file_id,
        'image_b64': image_b64
    }

    try:
        os.makedirs(f"/3d-images/spoof", exist_ok=True)
        os.makedirs(f"/3d-images/real", exist_ok=True)

        async with aiohttp.ClientSession() as session:
            async with session.post(vision_url, json=payload) as response:
                if response.status != 200:
                    raise HTTPException(status_code=404, detail="Error in vision service")
                response_data = await response.json()
                liveness_result = response_data.get("liveness_result")
                is_real = liveness_result.get("is_real")
                parent_path = "real" if is_real else "spoof"

        image_path = f"/3d-images/{parent_path}/{file_id}.jpeg"

        if os.path.exists(image_path):
            os.remove(image_path)

        async with aiofiles.open(image_path, "wb") as out_file:
            content = base64_2_bytestr(image_b64)
            await out_file.write(content)

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    return SiliconSaveResponse(is_real=is_real, file_id=file_id, liveness_result=liveness_result)