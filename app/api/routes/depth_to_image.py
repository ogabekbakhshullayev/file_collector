import io
import json

from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from typing import Annotated
import cv2
import numpy as np
from starlette.responses import StreamingResponse

router = APIRouter()


@router.post("/depth-to-image", response_class=StreamingResponse)
async def depth_to_image(
    json_depth_data: Annotated[UploadFile, "application/json"] = File(...),
    width: Annotated[int, "width of the image"] = Body(default=640),
    height: Annotated[int, "height of the image"] = Body(default=480),
):
    if not json_depth_data:
        raise HTTPException(status_code=404, detail="Files not found!")

    try:
        content = await json_depth_data.read()
        depth_data = np.array(json.loads(content).get("distance"))

        depth_data = depth_data.reshape((height, width))
        depth_image = depth_data.astype(np.uint8)

        depth_norm = cv2.normalize(
            depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        _, buffer = cv2.imencode(".png", depth_colormap)

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
