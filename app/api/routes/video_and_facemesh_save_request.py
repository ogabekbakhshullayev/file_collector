from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Body

from models.prediction import ResponseData
import aiofiles
from typing import Annotated

from uuid import uuid4
import os

router = APIRouter()


@router.post("/video-facemesh", response_model=ResponseData)
async def predict(
    near_image: Annotated[UploadFile, "image/jpg"] = File(...),
    far_image: Annotated[UploadFile, "image/jpg"] = File(...),
    facemesh: Annotated[UploadFile, "application/json"] = File(...),
    video: Annotated[UploadFile, "video/mp4"] = File(...),
    file_id: str = Body(...)
):
    if not near_image or not far_image or not facemesh or not video:
        raise HTTPException(status_code=404, detail="Files not found!")

    try:
        os.makedirs(f"/images-and-videos/near_image", exist_ok=True)
        os.makedirs(f"/images-and-videos/far_image", exist_ok=True)
        os.makedirs(f"/images-and-videos/facemesh", exist_ok=True)
        os.makedirs(f"/images-and-videos/video", exist_ok=True)

        # check if file_id already exists remove the files and save it again
        if os.path.exists(f"/images-and-videos/near_image/{file_id}.jpg"):
            os.remove(f"/images-and-videos/near_image/{file_id}.jpg")

        if os.path.exists(f"/images-and-videos/far_image/{file_id}.jpg"):
            os.remove(f"/images-and-videos/far_image/{file_id}.jpg")

        if os.path.exists(f"/images-and-videos/facemesh/{file_id}.json"):
            os.remove(f"/images-and-videos/facemesh/{file_id}.json")
        
        if os.path.exists(f"/images-and-videos/video/{file_id}.mp4"):
            os.remove(f"/images-and-videos/video/{file_id}.mp4")
            
        # Save near image
        near_image_path = f"/images-and-videos/near_image/{file_id}.jpg"
        async with aiofiles.open(near_image_path, "wb") as out_file:
            content = await near_image.read()
            await out_file.write(content)

        # Save far image
        far_image_path = f"/images-and-videos/far_image/{file_id}.jpg"
        async with aiofiles.open(far_image_path, "wb") as out_file:
            content = await far_image.read()
            await out_file.write(content)

        # Save facemesh
        facemesh_path = f"/images-and-videos/facemesh/{file_id}.json"
        async with aiofiles.open(facemesh_path, "wb") as out_file:
            content = await facemesh.read()
            await out_file.write(content)
        
        # Save video
        video_path = f"/images-and-videos/video/{file_id}.mp4"
        async with aiofiles.open(video_path, "wb") as out_file:
            content = await video.read()
            await out_file.write(content)
    
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")
    
    return ResponseData(
        status_code=200,
        file_id=file_id,
        detail="Near image, far image, facemesh and video saved successfully!"
    )