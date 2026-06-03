from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import aiofiles
import asyncio
import base64
import json
import os
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

SAVE_DIR = "/color-images"
_executor = ThreadPoolExecutor(max_workers=4)


class ColorImageItem(BaseModel):
    photo: str
    color: str


def _decode_base64(image_str: str) -> bytes:
    if "base64," in image_str:
        image_str = image_str.split("base64,")[1]
    padding = 4 - len(image_str) % 4
    if padding != 4:
        image_str += "=" * padding
    return base64.urlsafe_b64decode(image_str)


def _decode_all(items: list[dict]) -> list[tuple[str, bytes]]:
    result = []
    for item in items:
        color = item["color"].strip().lower()
        image_bytes = _decode_base64(item["photo"])
        result.append((color, image_bytes))
    return result


async def _write_file(path: str, data: bytes):
    async with aiofiles.open(path, "wb") as f:
        await f.write(data)


@router.post("/color-image-save")
@router.post("/color-image-save/", include_in_schema=False)
async def save_color_images(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    try:
        items_data = json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e}")

    if not isinstance(items_data, list) or len(items_data) == 0:
        raise HTTPException(status_code=400, detail="Expected non-empty list")

    # base64 decode is CPU-bound — run in thread pool
    loop = asyncio.get_event_loop()
    try:
        decoded = await loop.run_in_executor(_executor, _decode_all, items_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to decode images: {e}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session_dir = os.path.join(SAVE_DIR, timestamp)

    try:
        os.makedirs(session_dir, exist_ok=True)

        color_counts: dict[str, int] = {}
        tasks = []
        saved_files = []

        for color, image_bytes in decoded:
            count = color_counts.get(color, 0)
            color_counts[color] = count + 1
            filename = f"{color}.jpeg" if count == 0 else f"{color}_{count}.jpeg"
            saved_files.append(filename)
            tasks.append(_write_file(os.path.join(session_dir, filename), image_bytes))

        meta = {
            "timestamp": timestamp,
            "total_images": len(decoded),
            "colors": list(color_counts.keys()),
            "files": saved_files,
        }
        tasks.append(_write_file(
            os.path.join(session_dir, "request.json"),
            json.dumps(meta, ensure_ascii=False, indent=2).encode()
        ))

        await asyncio.gather(*tasks)

    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save: {e}")

    return {"session": timestamp, "colors": list(color_counts.keys()), "total_images": len(decoded)}
