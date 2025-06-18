import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os
from contextlib import asynccontextmanager

from config.settings import settings
from api.endpoints import router as api_router
from services.database import init_db

# --- Logging Configuration ---
log_level_str = settings.log_level.upper()
numeric_log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=numeric_log_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger_main = logging.getLogger("EidoSentinelAPIMain")
logger_main.info(f"API log level set to: {log_level_str}")

# --- Lifespan event for DB initialization ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger_main.info("FastAPI application startup...")
    await init_db()
    logger_main.info("Database initialization complete.")
    yield
    logger_main.info("FastAPI application shutdown.")

# --- FastAPI Application Instance ---
app = FastAPI(
    title="EIDO Sentinel API",
    description="API for ingesting EIDO reports, managing emergency incidents, and serving a showcase landing page.",
    version="0.9.1",
    contact={"name": "EIDO Sentinel Support",
             "url": "https://github.com/LXString/eido-sentinel"},
    license_info={"name": "MIT License",
                  "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan
)

# --- CORS Middleware Configuration ---
# This is a more robust configuration for allowing your deployed Streamlit frontend
# and local development environments to communicate with this backend.
allowed_origins = []

# 1. Add local development origins
local_streamlit_url = f"http://localhost:{settings.streamlit_server_port}"
# For the landing page JS
local_api_url = f"http://localhost:{settings.api_port}"
allowed_origins.extend([local_streamlit_url, local_api_url, "http://127.0.0.1:8501"])

# 2. Add deployed frontend URL from environment variables
# For deployment, you will set STREAMLIT_APP_URL in your backend's hosting environment (e.g., Fly.io secrets).
# Example: STREAMLIT_APP_URL="https://your-streamlit-app-name.streamlit.app"
streamlit_app_url = os.environ.get("STREAMLIT_APP_URL")
if streamlit_app_url:
    allowed_origins.append(streamlit_app_url)
    logger_main.info(
        f"Allowing CORS for deployed Streamlit app: {streamlit_app_url}")

# 3. Add the deployed backend's own URL if it's different from localhost
if "localhost" not in settings.api_base_url:
    allowed_origins.append(settings.api_base_url)

# Use "*" for development if you face persistent issues, but be specific for production.
# allowed_origins = ["*"] # <-- Uncomment for quick local testing if needed

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
logger_main.info(
    f"CORS middleware configured. Allowed origins: {allowed_origins}")

# --- Static Files Mounting ---
PROJECT_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(PROJECT_ROOT_DIR, "static")

if not os.path.isdir(STATIC_DIR):
    logger_main.warning(
        f"Static directory not found at: {STATIC_DIR}. Landing page may not be served.")
else:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger_main.info(f"Mounted static files from directory: {STATIC_DIR}")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_index(request: Request):
    index_html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_html_path):
        return FileResponse(index_html_path, media_type="text/html")
    else:
        logger_main.error(f"index.html not found in {STATIC_DIR}")
        return HTMLResponse(content="<h1>EIDO Sentinel API is running.</h1><p>Landing page (index.html) not found.</p>", status_code=200)

app.include_router(api_router)
logger_main.info("API router included at prefix /api/v1.")

if __name__ == "__main__":
    # For containerized deployments like Fly.io, the host must be "0.0.0.0".
    # We default to "0.0.0.0" which is safe for production and generally ok for local dev.
    # The PORT is read from the environment, which is standard for hosting platforms.
    uvicorn_host = os.getenv("HOST", "0.0.0.0")
    uvicorn_port = int(os.getenv("PORT", settings.api_port))

    logger_main.info(
        f"Preparing to start Uvicorn server on {uvicorn_host}:{uvicorn_port}")

    uvicorn.run(
        "api.main:app",
        host=uvicorn_host,
        port=uvicorn_port,
        log_level=settings.log_level.lower(),
        # Reload should be disabled in production. Checking for common prod env vars.
        reload=not (os.getenv("FLY_APP_NAME") or os.getenv("RENDER"))
    )