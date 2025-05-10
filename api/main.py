import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os

from config.settings import settings
from api.endpoints import router as api_router

# --- Logging Configuration ---
log_level_str = settings.log_level.upper()
numeric_log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=numeric_log_level, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True 
)

logger = logging.getLogger("EidoSentinelAPIMain")
logger.setLevel(numeric_log_level)
logger.info(f"API log level set to: {log_level_str}")


# --- FastAPI Application Instance ---
app = FastAPI(
    title="EIDO Sentinel API & Showcase",
    description="API for ingesting EIDO reports, managing emergency incidents, and showcase landing page.",
    version="0.8.1", 
    contact={"name": "EIDO Sentinel Support", "url": "https://github.com/LXString/eido-sentinel"},
    license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
)

# --- Middleware (CORS) ---
streamlit_port_setting = settings.streamlit_server_port
streamlit_port = streamlit_port_setting if isinstance(streamlit_port_setting, int) and streamlit_port_setting > 0 else 8501

origins = [
    "http://localhost", 
    f"http://localhost:{streamlit_port}",
    f"http://127.0.0.1:{streamlit_port}",
    "http://localhost:8000", # For the landing page itself if it makes API calls to itself
    "http://127.0.0.1:8000",
    # Add any other origins that might access the API directly
]

# Allow all origins if running in a flexible dev environment, or be more specific for production
# For this project, allowing localhost on common ports is reasonable.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Or ["*"] for testing, but be specific in production
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
logger.info(f"CORS middleware added. Allowed origins: {origins}")


# --- Static Files Mounting ---
# Get the absolute path to the project root directory
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(PROJECT_ROOT_DIR, "static")

# Ensure static directory exists
if not os.path.isdir(STATIC_DIR):
    logger.error(f"Static directory not found at: {STATIC_DIR}. Landing page will not be served.")
else:
    # Mount static files (CSS, JS, images)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"Mounted static files from directory: {STATIC_DIR}")

    # --- Root Endpoint to serve index.html (Landing Page) ---
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def read_index(request: Request):
        index_html_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_html_path):
            logger.debug(f"Serving index.html from {index_html_path}")
            return FileResponse(index_html_path, media_type="text/html")
        else:
            logger.error(f"index.html not found in {STATIC_DIR}")
            return HTMLResponse(content="<h1>EIDO Sentinel Landing Page Not Found</h1><p>Please check server configuration. Expected at static/index.html</p>", status_code=404)

# --- Include API Routers ---
# API routes will be under /api/v1
app.include_router(api_router) 
logger.info("API router included (expected at /api/v1).")


# --- Main execution block to run with Uvicorn ---
if __name__ == "__main__":
    api_host_setting = settings.api_host
    api_port_setting = settings.api_port

    api_host = api_host_setting if isinstance(api_host_setting, str) and api_host_setting else "127.0.0.1"
    api_port = api_port_setting if isinstance(api_port_setting, int) and api_port_setting > 0 else 8000
    
    logger.info(f"Starting Uvicorn server on {api_host}:{api_port}")
    logger.info(f"Landing page will be at http://{api_host}:{api_port}/")
    logger.info(f"API docs (Swagger UI) will be at http://{api_host}:{api_port}/docs")
    
    uvicorn.run(
        "api.main:app", 
        host=api_host,
        port=api_port,
        reload=True, 
        log_level=log_level_str.lower() 
    )