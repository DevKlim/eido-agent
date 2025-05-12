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

# --- CORS Configuration ---
# Add your deployed frontend URLs here.
# For local development with Streamlit on port 8503:
local_streamlit_url = f"http://localhost:{settings.streamlit_server_port}"
local_streamlit_url_ip = f"http://127.0.0.1:{settings.streamlit_server_port}"

# For deployed environments:
# Replace these with your actual URLs once deployed
render_backend_url = settings.api_base_url # This will be your *.onrender.com URL for the backend
streamlit_cloud_app_url = "https://your-streamlit-app-name.streamlit.app" # Replace with your Streamlit Cloud URL
github_pages_url = f"https://{os.environ.get('GITHUB_REPOSITORY_OWNER','your-username')}.github.io" # If serving static landing page separately

origins = [
    "http://localhost",             # General localhost
    local_streamlit_url,            # Local Streamlit (e.g., http://localhost:8503)
    local_streamlit_url_ip,         # Local Streamlit IP (e.g., http://127.0.0.1:8503)
    f"http://localhost:{settings.api_port}", # Local FastAPI (itself for landing page)
    f"http://127.0.0.1:{settings.api_port}", # Local FastAPI IP
]

# Add deployed URLs if they are set and different from localhost
if render_backend_url and "localhost" not in render_backend_url:
    origins.append(render_backend_url)
if streamlit_cloud_app_url and "your-streamlit-app-name" not in streamlit_cloud_app_url: # Check for placeholder
    origins.append(streamlit_cloud_app_url)
if github_pages_url and "your-username" not in github_pages_url: # Check for placeholder
    origins.append(github_pages_url)
# Remove duplicates
origins = sorted(list(set(origins)))


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
logger_main.info(f"CORS middleware added. Allowed origins (ensure placeholders like 'your-streamlit-app-name' are updated for deployment): {origins}")


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