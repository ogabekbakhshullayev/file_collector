from fastapi import APIRouter

from api.routes import file_save_request, depth_to_image

router = APIRouter()
router.include_router(file_save_request.router, tags=["file-saver"], prefix="/v1")
router.include_router(depth_to_image.router, tags=["depth-to-image"], prefix="/v1")
