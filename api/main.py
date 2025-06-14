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
logging.basicConfig(level=numeric_log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
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
    contact={"name": "EIDO Sentinel Support", "url": "https://github.com/LXString/eido-sentinel"},
    license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan
)

# --- CORS Middleware Configuration ---
# This is crucial for allowing your deployed Streamlit frontend to communicate with this backend.
origins = [
    # For local development
    f"http://localhost:{settings.streamlit_server_port}",
    f"http://127.0.0.1:{settings.streamlit_server_port}",
]

# For deployment, the Streamlit app URL should be added.
# It's best practice to set this via an environment variable in your backend's deployment environment.
# Example: STREAMLIT_APP_URL="https://your-app-name.streamlit.app"
streamlit_app_url = os.environ.get("STREAMLIT_APP_URL")
if streamlit_app_url:
    origins.append(streamlit_app_url)
    logger_main.info(f"Allowing CORS for deployed Streamlit app: {streamlit_app_url}")

# The backend API's own URL should also be an allowed origin if the landing page makes requests.
if "localhost" not in settings.api_base_url:
    origins.append(settings.api_base_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger_main.info(f"CORS middleware configured. Allowed origins: {origins}")

# --- Static Files Mounting ---
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(PROJECT_ROOT_DIR, "static")

if not os.path.isdir(STATIC_DIR):
    logger_main.warning(f"Static directory not found at: {STATIC_DIR}. Landing page may not be served.")
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
    uvicorn_host = os.getenv("HOST", settings.api_host)
    uvicorn_port = int(os.getenv("PORT", str(settings.api_port)))
    
    logger_main.info(f"Starting Uvicorn server on {uvicorn_host}:{uvicorn_port}")
    uvicorn.run(
        "api.main:app",
        host=uvicorn_host,
        port=uvicorn_port,
        reload=True, # Set to False in production
        log_level=log_level_str.lower()
    )