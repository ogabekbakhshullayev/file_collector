from fastapi import APIRouter

from api.routes import file_save_request, depth_to_image, near_far_facemesh_save_request, video_and_facemesh_save_request

router = APIRouter()
router.include_router(file_save_request.router, tags=["file-saver"], prefix="/v1")
router.include_router(depth_to_image.router, tags=["depth-to-image"], prefix="/v1")
router.include_router(near_far_facemesh_save_request.router, tags=["near-far-facemesh"], prefix="/v1")
router.include_router(video_and_facemesh_save_request.router, tags=["video-facemesh"], prefix="/v1")
