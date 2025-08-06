from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from api.routes.api import router as api_router
from core.events import create_start_app_handler
from core.config import API_PREFIX, DEBUG, PROJECT_NAME, VERSION


def get_application() -> FastAPI:
    application = FastAPI(title=PROJECT_NAME, debug=DEBUG, version=VERSION)
    application.include_router(api_router, prefix=API_PREFIX)
    
    # Add static files serving for test interface
    application.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Add homepage with available endpoints
    @application.get("/", response_class=HTMLResponse)
    async def homepage():
        """Show available endpoints and links"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>File Collector API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .method { background: #007bff; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
                .test-link { background: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }
                .test-link:hover { background: #218838; }
            </style>
        </head>
        <body>
            <h1>🚀 File Collector API</h1>
            <p>Welcome to the File Collector API. Here are the available endpoints:</p>
            
            <div class="endpoint">
                <h3>🛡️ Android Fraud Detection</h3>
                <p><span class="method">POST</span> <code>/api/v1/android-fraud-detection/analyze-device</code> - Analyze device metadata</p>
                <p><span class="method">POST</span> <code>/api/v1/android-fraud-detection/analyze-device-json</code> - Analyze from JSON file</p>
                <p><span class="method">POST</span> <code>/api/v1/android-fraud-detection/batch-analyze</code> - Batch analyze multiple files</p>
                <a href="/test" class="test-link">🧪 Test Multiple File Upload</a>
            </div>
            
            <div class="endpoint">
                <h3>📚 API Documentation</h3>
                <p><a href="/docs">📖 Interactive API Documentation (Swagger UI)</a></p>
                <p><a href="/redoc">📋 Alternative Documentation (ReDoc)</a></p>
            </div>
            
            <div class="endpoint">
                <h3>🏥 Health Check</h3>
                <p><span class="method">GET</span> <code>/api/v1/healthcheck</code> - Service health status</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    # Add test interface route
    @application.get("/test", response_class=HTMLResponse)
    async def test_interface():
        """Serve the multiple file upload test interface"""
        with open("static/test_multiple_files.html", "r") as f:
            return HTMLResponse(content=f.read())
    
    pre_load = False
    if pre_load:
        application.add_event_handler("startup", create_start_app_handler(application))
    return application


app = get_application()
