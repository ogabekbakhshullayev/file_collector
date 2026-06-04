from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import aiofiles
import json
import os
from uuid import uuid4

router = APIRouter()

BASE_DIR = "/camera-inject-data"


class CameraDataRequest(BaseModel):
    flag: str
    data: Dict[str, Any]
    session_id: Optional[str] = None


@router.post("/collect")
async def collect_camera_data(request: CameraDataRequest):
    """
    Collect camera parameter data labeled by flag (e.g. 'real', 'spoof', 'camera_inject').

    Saves the JSON payload under:
        /camera-inject-data/<flag>/<uuid>.json

    The flag value is dynamic — any string is accepted, and the corresponding
    subfolder is created automatically. Use consistent flag names so that later
    training can treat each subfolder as a class label.

    Body fields:
        flag        – label/class for this sample (e.g. "real", "spoof")
        data        – camera parameter dict (brightness, iso, exposure, etc.)
        session_id  – optional client-side session identifier stored in the file
    """
    flag = request.flag.strip()
    if not flag:
        raise HTTPException(status_code=400, detail="flag must not be empty")

    # Prevent path traversal
    if "/" in flag or "\\" in flag or ".." in flag:
        raise HTTPException(status_code=400, detail="flag contains invalid characters")

    folder = os.path.join(BASE_DIR, flag)
    os.makedirs(folder, exist_ok=True)

    file_id = str(uuid4())
    file_path = os.path.join(folder, f"{file_id}.json")

    payload = {
        "file_id": file_id,
        "flag": flag,
        "session_id": request.session_id,
        "data": request.data,
    }

    try:
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {err}")

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "file_id": file_id,
            "flag": flag,
            "saved_to": file_path,
        },
    )
