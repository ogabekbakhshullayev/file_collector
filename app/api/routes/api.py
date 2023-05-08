from fastapi import APIRouter

from api.routes import saver, depth_to_image

router = APIRouter()
router.include_router(saver.router, tags=["file-saver"], prefix="/v1")
router.include_router(depth_to_image.router, tags=["depth-to-image"], prefix="/v1")
