from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Body

from models.prediction import ResponseData
import aiofiles
from typing import Annotated

import os

router = APIRouter()


@router.post("/near_far_facemesh_save_request", response_model=ResponseData)
async def facemesh_and_image_save(
    near_image: Annotated[UploadFile, File(description="image/jpg")],
    far_image: Annotated[UploadFile, File(description="image/jpg")],
    facemesh: Annotated[UploadFile, File(description="application/json")],
    file_id: Annotated[str, Form()]
):
    if not near_image or not far_image or not facemesh:
        raise HTTPException(status_code=404, detail="Files not found!")
    

    try:
        os.makedirs(f"/images/near_image", exist_ok=True)
        os.makedirs(f"/images/far_image", exist_ok=True)
        os.makedirs(f"/images/facemesh", exist_ok=True)

        # check if file_id already exists remove the files and save it again
        if os.path.exists(f"/images/near_image/{file_id}.jpg"):
            os.remove(f"/images/near_image/{file_id}.jpg")

        if os.path.exists(f"/images/far_image/{file_id}.jpg"):
            os.remove(f"/images/far_image/{file_id}.jpg")

        if os.path.exists(f"/images/facemesh/{file_id}.json"):
            os.remove(f"/images/facemesh/{file_id}.json")
            
        # Save near image
        near_image_path = f"/images/near_image/{file_id}.jpg"
        async with aiofiles.open(near_image_path, "wb") as out_file:
            content = await near_image.read()
            await out_file.write(content)

        # Save far image
        far_image_path = f"/images/far_image/{file_id}.jpg"
        async with aiofiles.open(far_image_path, "wb") as out_file:
            content = await far_image.read()
            await out_file.write(content)

        # Save facemesh
        facemesh_path = f"/images/facemesh/{file_id}.json"
        async with aiofiles.open(facemesh_path, "wb") as out_file:
            content = await facemesh.read()
            await out_file.write(content)
    
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")
    
    return ResponseData(
        status_code=200,
        file_id=file_id,
        detail="File saved successfully!",
    )
    