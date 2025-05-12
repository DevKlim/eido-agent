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
from services.database import init_db # Import init_db

# --- Logging Configuration ---
log_level_str = settings.log_level.upper()
numeric_log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=numeric_log_level, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True 
)

logger_main = logging.getLogger("EidoSentinelAPIMain") # Renamed logger instance
logger_main.setLevel(numeric_log_level)
logger_main.info(f"API log level set to: {log_level_str}")


# --- Lifespan event for DB initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger_main.info("FastAPI application startup...")
    await init_db() # Initialize database and create tables
    logger_main.info("Database initialization complete.")
    yield
    logger_main.info("FastAPI application shutdown.")

# --- FastAPI Application Instance ---
app = FastAPI(
    title="EIDO Sentinel API",
    description="API for ingesting EIDO reports, managing emergency incidents, and showcase landing page.",
    version="0.9.1", # Incremented version
    contact={"name": "EIDO Sentinel Support", "url": "https://github.com/LXString/eido-sentinel"},
    license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan # Add lifespan context manager
)

# --- Middleware (CORS) ---
# Streamlit Cloud apps usually have a unique subdomain like your-app-name.streamlit.app
# GitHub Pages will be like your-username.github.io/your-repo-name/
# Render apps are typically your-app-name.onrender.com

streamlit_app_url_placeholder = "https://your-streamlit-app-name.streamlit.app" # Replace with actual
github_pages_url_placeholder = f"https://{os.environ.get('GITHUB_REPOSITORY_OWNER','your-username')}.github.io" # Replace with actual or derive
# If your GH Pages site is for the repo, it might be /your-repo-name

origins = [
    "http://localhost", # General localhost
    f"http://localhost:{settings.streamlit_server_port}", # Local Streamlit
    f"http://127.0.0.1:{settings.streamlit_server_port}", # Local Streamlit
    "http://localhost:8000", # Local FastAPI (itself for landing page)
    "http://127.0.0.1:8000", # Local FastAPI
    # Add deployed frontend URLs here once known:
    streamlit_app_url_placeholder, 
    github_pages_url_placeholder,
    # If your Render backend has a custom domain, add that too.
    # Example: "https://api.eidosentinel.com"
]
if settings.api_base_url and settings.api_base_url not in origins: # Add the deployed API_BASE_URL itself if distinct
    origins.append(settings.api_base_url)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
logger_main.info(f"CORS middleware added. Allowed origins (ensure placeholders are updated for deployment): {origins}")


# --- Static Files Mounting ---
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(PROJECT_ROOT_DIR, "static")

if not os.path.isdir(STATIC_DIR):
    logger_main.error(f"Static directory not found at: {STATIC_DIR}. Landing page will not be served.")
    try:
        os.makedirs(STATIC_DIR, exist_ok=True)
        placeholder_html_path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(placeholder_html_path):
            with open(placeholder_html_path, "w") as f:
                f.write("<h1>EIDO Sentinel Placeholder</h1><p>Landing page content is being generated.</p>")
            logger_main.info(f"Created placeholder index.html at {placeholder_html_path}")
    except Exception as e:
        logger_main.error(f"Could not create static directory or placeholder index.html: {e}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
logger_main.info(f"Mounted static files from directory: {STATIC_DIR}")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_index(request: Request):
    index_html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_html_path):
        logger_main.debug(f"Serving index.html from {index_html_path}")
        return FileResponse(index_html_path, media_type="text/html")
    else:
        logger_main.error(f"index.html not found in {STATIC_DIR}")
        return HTMLResponse(content="<h1>EIDO Sentinel Landing Page Not Found</h1>", status_code=404)

app.include_router(api_router) 
logger_main.info("API router included (expected at /api/v1).")

if __name__ == "__main__":
    # For local dev, Uvicorn uses these settings.
    # For Render, PORT is set by Render, HOST should be 0.0.0.0.
    # The Procfile will specify `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
    
    uvicorn_host = os.getenv("HOST", settings.api_host) # Render sets HOST to 0.0.0.0
    uvicorn_port = int(os.getenv("PORT", str(settings.api_port))) # Render sets PORT

    logger_main.info(f"Starting Uvicorn server on {uvicorn_host}:{uvicorn_port}")
    uvicorn.run(
        "api.main:app", 
        host=uvicorn_host,
        port=uvicorn_port,
        reload=True, # Reload should be False for production on Render typically
        log_level=log_level_str.lower() 
    )